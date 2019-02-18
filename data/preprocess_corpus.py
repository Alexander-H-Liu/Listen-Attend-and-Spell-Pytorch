import sys
sys.path.insert(0, '..')
from src.preprocess import extract_feature,encode_target
from joblib import Parallel, delayed
import argparse
import os 
import csv
from pathlib import Path
from os.path import join
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd

"""
<data_path>(corpus name)
├── <train>
|   ├── audio
|   |   ├── <1.mp3>
|   |   └── <2.mp3>
|   └── transcript.csv
└── <test>
    ├── audio
    |   ├── <1.mp3>
    |   └── <2.mp3>
    └── transcript.csv

In transcript.csv...
audio,text
<1.mp3>,<blablabla>
<2.mp3>,<blabla>
"""
def change_fe(cur_path, chg_dir, file_extension):
    """ change file extension """
    _, file_name = os.path.split(cur_path)
    base = os.path.splitext(file_name)[0]
    return join(chg_dir, base+file_extension)

def bpe(paras, corpus, dim, sets, bpe_dir):
    # Select dataset to train bpe
    for idx, s in enumerate(sets):
        print('\t', idx, ':', s)
    bpe_tr = input('Please enter the indices of training sets for BPE training (seperate w/ space): ')
    bpe_tr = [sets[int(t)] for t in bpe_tr.split(' ')]

    # Collect text
    tr_txt = []
    for s in bpe_tr:
        with open(join(paras.data_path, s, 'transcript.csv'), 'r') as f:
            rows = csv.reader(f)
            texts = list(zip(*rows))[1][1:]
            tr_txt += texts
    with open(join(bpe_dir, 'transcript.txt'), 'w') as tf:
        for text in tr_txt:
            tf.write(text + '\n')

    # Train BPE
    from subprocess import call
    call(['spm_train',
          '--input=' + os.path.join(bpe_dir, 'transcript.txt'),
          '--model_prefix=' + os.path.join(bpe_dir, 'bpe'),
          '--vocab_size=' + str(paras.n_tokens),
          '--character_coverage=1.0'
        ])
    # Encode data
    for s in sets:
        with open(join(paras.data_path, s, 'transcript.csv'), 'r') as f:
            rows = csv.reader(f)
            texts = list(zip(*rows))[1][1:]
            with open(join(bpe_dir,'raw',s+'.txt'), 'w') as f:
                for sent in texts: f.write(sent+'\n')
            call(['spm_encode',
                  '--model='+os.path.join(bpe_dir,'bpe.model'),
                  '--output_format=piece'
                ],stdin=open(join(bpe_dir,'raw',s+'.txt'),'r'),
                  stdout=open(join(bpe_dir,'encode',s+'.txt'),'w'))

    # Make Dict
    encode_table = {'<sos>':0,'<eos>':1}
    with open(join(bpe_dir,'bpe.vocab'), 'r', encoding="utf-8") as f:
        for line in f:
            tok = line.split('\t')[0]
            if tok not in ['<s>', '</s>']:
                encode_table[tok] = len(encode_table)
    return encode_table

def feature_extract(paras, sets, output_dir, bpe_dir, encode_table):
    print('Data sets :')
    for idx, s in enumerate(sets):
        print('\t', idx, ':', s)
    tr_set = input('Please enter the index of splits you wish to use preprocess. (seperate with space): ')
    tr_set = [sets[int(t)] for t in tr_set.split(' ')]
    for s in tr_set:
        audio_dir = join(paras.data_path, s, 'audio')
        todos = []
        with open(join(paras.data_path, s, 'transcript.csv'), 'r') as f:
            rows = csv.reader(f)
            todos = list(zip(*rows))[0][1:]
            todos = [join(audio_dir, f) for f in todos]
        print('Encoding target...', flush=True)
        tr_y = []
        with open(join(bpe_dir, 'encode', s+'.txt'), 'r') as f:
            for line in f: tr_y.append(line[:-1].split(' '))
        tr_y, encode_table = encode_target(tr_y, table=encode_table, mode='subword', max_idx=paras.n_tokens)
        cur_path = os.path.join(output_dir, s)
        if not os.path.exists(cur_path): os.makedirs(cur_path)
        print('Extracting acoustic feature...{}'.format(s))
        tr_x = Parallel(n_jobs=paras.n_jobs)(delayed(extract_feature)(str(file_path),feature=paras.feature_type,dim=paras.feature_dim,\
                    cmvn=paras.apply_cmvn,delta=paras.apply_delta,delta_delta=paras.apply_delta_delta,\
                    save_feature=change_fe(file_path, cur_path, '')) for file_path in tqdm(todos))
        #TODO: test
        # def foo(file_path):
            # return len(np.load(file_path))
        # tr_x = Parallel(n_jobs=paras.n_jobs)(delayed(foo)(change_fe(file_path, cur_path, '.npy')) for file_path in tqdm(todos))
        # sort by len
        sorted_idx = list(reversed(np.argsort(tr_x)))
        sorted_y = ['_'.join([str(i) for i in tr_y[idx]]) for idx in sorted_idx]
        sorted_todos = [change_fe(str(todos[idx]), s, '.npy') for idx in sorted_idx]
        # Dump label
        df = pd.DataFrame(data={'file_path':[fp for fp in sorted_todos],'length':list(reversed(sorted(tr_x))),'label':sorted_y})
        df.to_csv(join(output_dir, s+'.csv'))

        with open(join(output_dir,"mapping.pkl"), "wb") as fp:
            pickle.dump(encode_table, fp)

    print('All done, saved at', output_dir, 'exit.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess program for Corpus.')
    parser.add_argument('--data_path', type=str, help='List of path to raw dataset')
    parser.add_argument('--feature_type', default='fbank', type=str, help='Feature type ( mfcc / fbank )', required=False)
    parser.add_argument('--feature_dim', default=40, type=int, help='Dimension of feature', required=False)
    parser.add_argument('--apply_delta', default=True, type=bool, help='Append Delta', required=False)
    parser.add_argument('--apply_delta_delta', default=False, type=bool, help='Append Delta Delta', required=False)
    parser.add_argument('--apply_cmvn', default=True, type=bool, help='Apply CMVN on feature', required=False)
    parser.add_argument('--output_path', default='.', type=str, help='Path to store output', required=False)
    parser.add_argument('--n_jobs', default=-1, type=int, help='Number of jobs used for feature extraction', required=False)
    parser.add_argument('--n_tokens', default=5000, type=int, help='Vocabulary size of target', required=False)
    paras = parser.parse_args()

    dim = paras.feature_dim * (1+paras.apply_delta+paras.apply_delta_delta)
    corpus = os.path.basename(os.path.normpath(paras.data_path))
    sets = [d for d in os.listdir(paras.data_path) if os.path.isdir(join(paras.data_path, d))]

    # Setup path
    output_dir = join(paras.output_path, '_'.join([corpus, str(paras.feature_type)+str(dim), 'subword'+str(paras.n_tokens)]))
    bpe_dir = join(output_dir, 'bpe')
    if not os.path.exists(bpe_dir):
        os.makedirs(join(bpe_dir, 'raw'))
        os.makedirs(join(bpe_dir, 'encode'))
    # BPE training
    encode_table = bpe(paras, corpus, dim, sets, bpe_dir)
    feature_extract(paras, sets, output_dir, bpe_dir, encode_table)
