import os
import pickle
import torch
import numpy as np
import editdistance as ed

class Mapper():
    '''Mapper for index2token'''
    def __init__(self,file_path):
        # Find mapping
        with open(os.path.join(file_path,'mapping.pkl'),'rb') as fp:
            self.mapping = pickle.load(fp)
        self.r_mapping = {v:k for k,v in self.mapping.items()}

    def get_dim(self):
        return len(self.mapping)

    def translate(self,seq,return_string=False):
        new_seq = []
        for c in seq:
            new_seq.append(self.r_mapping[c])
        if return_string:
            new_seq = ''.join(new_seq).replace('<sos>','').replace('<eos>','').replace('▁',' ').lstrip()
        return new_seq

    def idx2word(self,seq,metric):
        pred = [self.translate(trim_eos(p)) for p in seq]
        if metric =='per':
            pred = [collapse_phn(p) for p in pred]
        return pred

class Hypothesis:
    '''Hypothesis for beam search decoding.
       Stores the history of label sequence & score 
       Stores the previous decoder state, ctc state, ctc score, lm state and attention map (if necessary)'''
    
    def __init__(self, decoder_state, emb, output_seq=[], output_scores=[], 
                 lm_state=None, ctc_state=None,ctc_prob=0.0,att_map=None):
        assert len(output_seq) == len(output_scores)
        # attention decoder
        self.decoder_state = decoder_state
        self.att_map = att_map
        
        # RNN language model
        self.lm_state = lm_state
        
        # Previous outputs
        self.output_seq = output_seq
        self.output_scores = output_scores
        
        # CTC decoding
        self.ctc_state = ctc_state
        self.ctc_prob = ctc_prob
        
        # Embedding layer for last_char
        self.emb = emb
        

    def avgScore(self):
        '''Return the averaged log probability of hypothesis'''
        assert len(self.output_scores) != 0
        return sum(self.output_scores) / len(self.output_scores)

    def addTopk(self, topi, topv, decoder_state, att_map=None,
                lm_state=None, ctc_state=None, ctc_prob=0.0, ctc_candidates=[]):
        '''Expand current hypothesis with a given beam size'''
        new_hypothesis = []
        term_score = None
        ctc_s,ctc_p = None,None
        beam_size = len(topi[0])
        
        for i in range(beam_size):
            # Detect <eos>
            if topi[0][i].item() == 1:
                term_score = topv[0][i].cpu()
                continue
            
            idxes = self.output_seq[:]     # pass by value
            scores = self.output_scores[:] # pass by value
            idxes.append(topi[0][i].cpu())
            scores.append(topv[0][i].cpu()) 
            if ctc_state is not None:
                #idx = topi[0][i].item() #
                idx = ctc_candidates.index(topi[0][i].item()) #
                ctc_s = ctc_state[idx,:,:]
                ctc_p = ctc_prob[idx]
            new_hypothesis.append(Hypothesis(decoder_state, self.emb,
                                      output_seq=idxes, output_scores=scores, lm_state=lm_state,
                                      ctc_state=ctc_s,ctc_prob=ctc_p,att_map=att_map))
        if term_score is not None:
            self.output_seq.append(torch.tensor(1))
            self.output_scores.append(term_score)
            return self, new_hypothesis
        return None, new_hypothesis

    @property
    def outIndex(self):
        return [i.item() for i in self.output_seq]

    @property
    def last_char_idx(self):
        idx = self.output_seq[-1] if len(self.output_seq) != 0 else 0
        return torch.LongTensor([[idx]])
    @property
    def last_char(self):
        idx = self.output_seq[-1] if len(self.output_seq) != 0 else 0
        return self.emb(torch.LongTensor([idx]).to(next(self.emb.parameters()).device))

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

def cal_cer(pred,label,metric,mapper,get_sentence=False, argmax=True):
    if argmax:
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
