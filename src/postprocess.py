import os
import pickle
import torch
import numpy as np
import editdistance as ed

class Mapper():
    def __init__(self,file_path):
        # Find mapping
        with open(os.path.join(file_path,'mapping.pkl'),'rb') as fp:
            self.mapping = pickle.load(fp)
        self.r_mapping = {v:k for k,v in self.mapping.items()}

    def get_dim(self):
        return len(self.mapping)

    def translate(self,seq):
        new_seq = []
        for c in seq:
            new_seq.append(self.r_mapping[c])
        return new_seq

    def idx2word(self,seq,metric):
        pred = [self.translate(trim_eos(p)) for p in seq]
        if metric =='per':
            pred = [collapse_phn(p) for p in pred]
        return pred



def cal_acc(pred,label):
    pred = np.argmax(pred.cpu().detach(),axis=-1)
    label = label.cpu()
    accs = []
    for p,l in zip(pred,label):
        correct = 0.0
        total_char = 0
        for pp,ll in zip(p,l):
            if ll == 0: break
            correct += int(pp==ll)
            total_char += 1
        accs.append(correct/total_char)
    return sum(accs)/len(accs)

def cal_cer(pred,label,metric,mapper,get_sentence=False):
    pred = np.argmax(pred.cpu().detach(),axis=-1)
    label = label.cpu()
    pred = mapper.idx2word(pred,metric)
    label = mapper.idx2word(label,metric)
    if 'sub' in metric:
        pred = [''.join(p).replace('▁',' ') for p in pred]
        label = [''.join(l).replace('▁',' ') for l in label]
        if get_sentence:
            return pred,label
        if 'wer' in metric:
            pred = [p.split(' ') for p in pred]
            label = [l.split(' ') for l in label]
        elif 'cer' in metric:
            pred = [list(p) for p in pred]
            label = [list(l) for l in label]

    if get_sentence:
        sep = '' if metric == 'cer' or 'sub' in metric else ' '
        return [sep.join(p) for p in pred],[sep.join(l) for l in label]
    
    eds = [float(ed.eval(p,l))/len(l) for p,l in zip(pred,label)]
    
    return sum(eds)/len(eds)

# Only draw first attention head
def draw_att(att_list,hyp_txt):
    attmaps = []
    for att,hyp in zip(att_list[0],np.argmax(hyp_txt.cpu().detach(),axis=-1)):
        att_len = len(trim_eos(hyp))+1
        att = att.detach().cpu()
        attmaps.append(torch.stack([att,att,att],dim=0)[:,:att_len,:]) # +1 for att. @ <eos>
    return attmaps


def collapse_phn(seq):
    #phonemes = ["b", "bcl", "d", "dcl", "g", "gcl", "p", "pcl", "t", "tcl", "k", "kcl", "dx", "q", "jh", "ch", "s", "sh", "z", "zh", 
    #"f", "th", "v", "dh", "m", "n", "ng", "em", "en", "eng", "nx", "l", "r", "w", "y", 
    #"hh", "hv", "el", "iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay", "ah", "ao", "oy",
    #"ow", "uh", "uw", "ux", "er", "ax", "ix", "axr", "ax-h", "pau", "epi", "h#"]

    phonemse_reduce_mapping = {"b":"b", "bcl":"h#", "d":"d", "dcl":"h#", "g":"g", "gcl":"h#", "p":"p", "pcl":"h#", "t":"t", "tcl":"h#", "k":"k", "kcl":"h#", "dx":"dx", "q":"q", "jh":"jh", "ch":"ch", "s":"s", "sh":"sh", "z":"z", "zh":"sh", 
    "f":"f", "th":"th", "v":"v", "dh":"dh", "m":"m", "n":"n", "ng":"ng", "em":"m", "en":"n", "eng":"ng", "nx":"n", "l":"l", "r":"r", "w":"w", "y":"y", 
    "hh":"hh", "hv":"hh", "el":"l", "iy":"iy", "ih":"ih", "eh":"eh", "ey":"ey", "ae":"ae", "aa":"aa", "aw":"aw", "ay":"ay", "ah":"ah", "ao":"aa", "oy":"oy",
    "ow":"ow", "uh":"uh", "uw":"uw", "ux":"uw", "er":"er", "ax":"ah", "ix":"ih", "axr":"er", "ax-h":"ah", "pau":"h#", "epi":"h#", "h#": "h#","<sos>":"<sos>","<unk>":"<unk>"}

    new_seq = []
    for c in seq:
        new_seq.append(phonemse_reduce_mapping[c])

    return new_seq

def trim_eos(seqence):
    new_pred = []
    for char in seqence:
        if char != 1:
            new_pred.append(int(char))
        else:
            break
    return new_pred
