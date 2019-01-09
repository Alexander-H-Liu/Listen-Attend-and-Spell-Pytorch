import sys
sys.path.insert(0, '..')
import argparse
import os 
import pickle
from tqdm import tqdm
import pandas as pd
from subprocess import call
import random

TOTAL_SENTENCES = 40418261

parser = argparse.ArgumentParser(description='Preprocess program for librispeech-lm-norm.txt')
parser.add_argument('--data_path', type=str, help='Path to LibriSpeech\'s extra text.')
parser.add_argument('--output_path',type=str, help='Path to store output csv file')
#parser.add_argument('--bpe_source', type=str, help='Path to pretrained BPE.')
parser.add_argument('--target', default='subword', type=str, help='Learning target ( phoneme / char / subword / word )', required=False)
parser.add_argument('--n_tokens', default=5000, type=int, help='Vocabulary size of target', required=False)
parser.add_argument('--n_samples', default=160000, type=int, help='Sample a part of txt as training data.', required=False)
paras = parser.parse_args()


# Encode data
with open(os.path.join(paras.output_path,"mapping.pkl"), "rb") as fp:
    word_dic = pickle.load(fp)
bpe_dir = os.path.join(paras.output_path,"bpe")

call(['spm_encode',
      '--model='+os.path.join(bpe_dir,'bpe.model'),
      '--output_format=piece'
    ],stdin=open(paras.data_path,'r'),
      stdout=open(os.path.join(bpe_dir,'encode','extra.txt'),'w'))
y = []

i = 0
freq = TOTAL_SENTENCES//paras.n_samples
with open(os.path.join(bpe_dir,'encode','extra.txt'),'r') as f:
    for line in f:
        i += 1
        if i%freq == 0:
            tmp = ['0']+[str(word_dic[c]) for c in line[:-1].split(' ')]+['1']
            y.append('_'.join(tmp))
df = pd.DataFrame(data={'file_path':['None' for f in y],'length':[0 for f in y],'label':y})
df.to_csv(os.path.join(paras.output_path,'extra.csv'))
