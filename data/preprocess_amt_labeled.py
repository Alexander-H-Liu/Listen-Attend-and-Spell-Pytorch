# Code for  labeled data (gender). This is used to test the model with acoustic classifier.
# label in this version refer to  gender and old label is renamed to transcription 
import sys
sys.path.insert(0, '..')
from src.preprocess import extract_feature,encode_target
from joblib import Parallel, delayed
import argparse
import os 
from pathlib import Path
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd

def boolean_string(s):
    if s not in ['False', 'True']:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser(description='Preprocess program for LibriSpeech dataset.')
parser.add_argument('--data_path', type=str, help='Path to raw files')
parser.add_argument('--input_segment_file', type=str, help='csv contains labels/transcripts and file path')
parser.add_argument('--feature_type', default='fbank', type=str, help='Feature type ( mfcc / fbank )', required=False)
parser.add_argument('--feature_dim', default=40, type=int, help='Dimension of feature', required=False)
parser.add_argument('--apply_delta', default=True, type=boolean_string, help='Append Delta', required=False)
parser.add_argument('--apply_delta_delta', default=False, type=boolean_string, help='Append Delta Delta', required=False)
parser.add_argument('--apply_cmvn', default=True, type=boolean_string, help='Apply CMVN on feature', required=False)
parser.add_argument('--output_path', default='.', type=str, help='Path to store output', required=False)
parser.add_argument('--n_jobs', default=-1, type=int, help='Number of jobs used for feature extraction', required=False)
parser.add_argument('--target', default='subword', type=str, help='Learning target ( phoneme / char / subword / word )', required=False)
parser.add_argument('--n_tokens', default=5000, type=int, help='Vocabulary size of target', required=False)
parser.add_argument("--label_type", type=str, help="gad/phq", required=True)
parser.add_argument("--libri_train", type=str, help="libri training data", required=False)
paras = parser.parse_args()



def read_text(text, target):
    #src_file = '-'.join(file.split('-')[:-1])+'.trans.txt'
    #idx = file.split('/')[-1].split('.')[0]
    # convert to upper
    text = text.upper()  # preprocessing
    if target =='char':
        return [c for c in text.split(' ',1)[1]]
    elif target =='subword':
        return text.split(' ',1)[1]
    else:
        raise ValueError('Unsupported target: '+target)



# Process data
sets = ['train','dev','dev2','test', 'libri']
encode_table = None
output_dir = None
dim = paras.feature_dim*(1+paras.apply_delta+paras.apply_delta_delta)


segments = pd.read_csv(paras.input_segment_file, sep="|")
fnames = segments["fname"]
labs = segments["labels"]
transcripts = segments["transcription"]
splits = segments["split"]
questions = segments["question"]

# read the speakers data 
# phq or gad
id2phqgad = {}
id2trans = {}
id2split = {}

for fi, la, tr, sp, q in zip(fnames, labs, transcripts, splits, questions):  # TODO  check make sure correct
    
    if q == "control":
        continue

    parts = la.split()
    if paras.label_type == "phq":
        l = float(parts[8].split("_")[1])
    elif paras.label_type == "gad":
        l = float(parts[16].split("_")[1])
    id2phqgad[fi] = l
    id2trans[fi] = tr
    id2split[fi] = sp


# BPE training
if paras.target == 'subword':
    # Setup path
    output_dir = os.path.join(paras.output_path,'_'.join(['amt',str(paras.feature_type)+str(dim),str(paras.target)+str(paras.n_tokens)]))
    if not os.path.exists(output_dir):os.makedirs(output_dir)
    bpe_dir = os.path.join(output_dir,'bpe')
    if not os.path.exists(bpe_dir):os.makedirs(bpe_dir)

    # Select dataset
    print('')
    print('Pretrain BPE for subword unit.')
    print('Data sets :')
    for idx,s in enumerate(sets):
        print('\t',idx,':',s)
    bpe_tr = input('Please enter the index for training sets for BPE (seperate w/ space): ')
    bpe_tr = [sets[int(t)] for t in bpe_tr.split(' ')]

    # Collect text
    tr_txt = []
    for s in bpe_tr:
        if s == "libri":
            continue
        #todo = list(Path(os.path.join(paras.data_path,s)).rglob("*.wav"))
        #tr_txt+=Parallel(n_jobs=paras.n_jobs)(delayed(read_text)(str(file),target=paras.target) for file in todo)
        for key in id2split:
            if  id2split[key] == s:
                tr_txt += read_text(id2trans[key], "subword")    
    with open(os.path.join(bpe_dir,'train.txt'),'w') as f:   # TODO Concat or load other data 
        for s in tr_txt:f.write(s+'\n')

        if "libri" in bpe_tr:
            lines = [line.strip() for line in open(paras.libri_train)]
            for line in lines:
                f.write(line + "\n")

    # Train BPE
    from subprocess import call
    call(['spm_train',
          '--input='+os.path.join(bpe_dir,'train.txt'),
          '--model_prefix='+os.path.join(bpe_dir,'bpe'),
          '--vocab_size='+str(paras.n_tokens),
          '--character_coverage=1.0'
        ])
    # Encode data
    if not os.path.exists(os.path.join(bpe_dir,'raw')):os.makedirs(os.path.join(bpe_dir,'raw'))
    if not os.path.exists(os.path.join(bpe_dir,'encode')):os.makedirs(os.path.join(bpe_dir,'encode'))
    txts = {}
    for s in sets:
        if s == "libri":
            continue
        for key in id2split:
            if  id2split[key] == s:
                if s not in txts:
                    txts[s] = []
                txts[s] += read_text(id2trans[key], "subword") 


    for s in txts:
        #todo = list(Path(os.path.join(paras.data_path,s)).rglob("*.flac"))
        #txts = Parallel(n_jobs=paras.n_jobs)(delayed(read_text)(str(file),target=paras.target) for file in todo)
        with open(os.path.join(bpe_dir,'raw',s+'.txt'),'w') as f:
            for sent in txts[s]:f.write(sent+'\n')
        call(['spm_encode',
              '--model='+os.path.join(bpe_dir,'bpe.model'),
              '--output_format=piece'
            ],stdin=open(os.path.join(bpe_dir,'raw',s+'.txt'),'r'),
              stdout=open(os.path.join(bpe_dir,'encode',s+'.txt'),'w'))

    # Make Dict
    encode_table = {'<sos>':0,'<eos>':1}
    with open(os.path.join(bpe_dir,'bpe.vocab'),'r', encoding="utf-8") as f:
        for line in f:
            tok = line.split('\t')[0]
            if tok not in ['<s>','</s>']:
                encode_table[tok] = len(encode_table)

print('')
print('Data sets :')
for idx,s in enumerate(sets):
    print('\t',idx,':',s)
tr_set = input('Please enter the index of splits you wish to use preprocess. (seperate with space): ')
tr_set = [sets[int(t)] for t in tr_set.split(' ')]


# Acoustic Feature Extraction & Make Date Table
for s in tr_set:
    print('')
    print('Preprocessing',s,'data...',end='')
    todo = list(Path(os.path.join(paras.data_path,s)).rglob("*.wav"))
    print(len(todo),'audio files found in',s)

    print('Encoding target...',flush=True)
    if paras.target == 'subword':
        tr_y = []
        with open(os.path.join(bpe_dir,'encode',s+'.txt'),'r') as f:
            for line in f:tr_y.append(line[:-1].split(' '))
    else:
        #tr_y = Parallel(n_jobs=paras.n_jobs)(delayed(read_text)(str(file),target=paras.target) for file in tqdm(todo))  #TODO  this line should change
        for key in id2split:
            if  id2split[key] == s:
                tr_y += read_text(id2trans[key], paras.target) 


    tr_y, encode_table = encode_target(tr_y,table=encode_table,mode=paras.target,max_idx=paras.n_tokens)  # ????

    if output_dir is None:
        output_dir = os.path.join(paras.output_path,'_'.join(['amt',str(paras.feature_type)+str(dim),str(paras.target)+str(len(encode_table))]))
    if not os.path.exists(output_dir):os.makedirs(output_dir)
    cur_path = os.path.join(output_dir,s)
    if not os.path.exists(cur_path):os.makedirs(cur_path)

    print('Extracting acoustic feature...',flush=True)
    tr_x = Parallel(n_jobs=paras.n_jobs)(delayed(extract_feature)(str(file),feature=paras.feature_type,dim=paras.feature_dim,\
            cmvn=paras.apply_cmvn,delta=paras.apply_delta,delta_delta=paras.apply_delta_delta,save_feature=os.path.join(cur_path,str(file).split('/')[-1].replace('.wav',''))) for file in tqdm(todo))

    # sort by len
    sorted_y = ['_'.join([str(i) for i in tr_y[idx]]) for idx in reversed(np.argsort(tr_x))]
    sorted_todo = [os.path.join(s,str(todo[idx]).split('/')[-1].replace('.wav','.npy')) for idx in reversed(np.argsort(tr_x))]
    
    spk_ids = [x.split("/")[1].split("-")[0] for x in sorted_todo] # TODO ???????
    sorted_labels = [id2phqgad[x] for x in spk_ids]

    # Dump label
    df = pd.DataFrame(data={'file_path':[fp for fp in sorted_todo],'length':list(reversed(sorted(tr_x))),'transcription':sorted_y, "label": sorted_labels})
    df.to_csv(os.path.join(output_dir,s+'.csv'))

    with open(os.path.join(output_dir,"mapping.pkl"), "wb") as fp:
        pickle.dump(encode_table, fp)

print('All done, saved at',output_dir,'exit.')
