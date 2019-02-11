import editdistance as ed
import pandas as pd
import argparse

# Arguments
parser = argparse.ArgumentParser(description='Evaluate decoding result.')
parser.add_argument('--file', type=str, help='Path to decode result file.')
paras = parser.parse_args()

                        
decode = pd.read_csv(paras.file,sep='\t',header=None)
truth = decode[0].tolist()
pred = decode[1].tolist()
cer = []
wer = []
for gt,pd in zip(truth,pred):
    wer.append(ed.eval(pd.split(' '),gt.split(' '))/len(gt.split(' ')))
    cer.append(ed.eval(pd,gt)/len(gt))

print('CER : {:.6f}'.format(sum(cer)/len(cer)))
print('WER : {:.6f}'.format(sum(wer)/len(wer)))