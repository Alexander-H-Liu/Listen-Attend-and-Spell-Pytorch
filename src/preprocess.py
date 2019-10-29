# Reference: https://groups.google.com/forum/#!msg/librosa/V4Z1HpTKn8Q/1-sMpjxjCSoJ

import librosa
import numpy as np
from operator import itemgetter
# NOTE: there are warnings for MFCC extraction due to librosa's issue
import warnings
warnings.filterwarnings("ignore")

# Acoustic Feature Extraction
# Parameters
#     - input file  : str, audio file path
#     - feature     : str, fbank or mfcc
#     - dim         : int, dimension of feature
#     - cmvn        : bool, apply CMVN on feature
#     - window_size : int, window size for FFT (ms)
#     - stride      : int, window stride for FFT
#     - save_feature: str, if given, store feature to the path and return len(feature)
# Return
#     acoustic features with shape (time step, dim)
def extract_feature(input_file,feature='fbank',dim=40, cmvn=True, delta=False, delta_delta=False,
                    window_size=25, stride=10,save_feature=None):
    y, sr = librosa.load(input_file,sr=None)
    ws = int(sr*0.001*window_size)
    st = int(sr*0.001*stride)
    if feature == 'fbank': # log-scaled
        feat = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=dim,
                                    n_fft=ws, hop_length=st)
        feat = np.log(feat+1e-6)
    elif feature == 'mfcc':
        feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=dim, n_mels=26,
                                    n_fft=ws, hop_length=st)
        feat[0] = librosa.feature.rmse(y, hop_length=st, frame_length=ws) 
        
    else:
        raise ValueError('Unsupported Acoustic Feature: '+feature)

    feat = [feat]
    if delta:
        feat.append(librosa.feature.delta(feat[0]))

    if delta_delta:
        feat.append(librosa.feature.delta(feat[0],order=2))
    feat = np.concatenate(feat,axis=0)
    if cmvn:
        feat = (feat - feat.mean(axis=1)[:,np.newaxis]) / (feat.std(axis=1)+1e-16)[:,np.newaxis]
    if save_feature is not None:
        tmp = np.swapaxes(feat,0,1).astype('float32')
        np.save(save_feature,tmp)
        return len(tmp)
    else:
        return np.swapaxes(feat,0,1).astype('float32')

# Target Encoding Function
# Parameters
#     - input list : list, list of target list
#     - table      : dict, token-index table for encoding (generate one if it's None)
#     - mode       : int, encoding mode ( phoneme / char / subword / word )
#     - max idx    : int, max encoding index (0=<sos>, 1=<eos>, 2=<unk>)
# Return
#     - output list: list, list of encoded targets
#     - output dic : dict, token-index table used during encoding
def encode_target(input_list,table=None,mode='subword',max_idx=500):
    if table is None:
        ### Step 1. Calculate wrd frequency
        table = {}
        for target in input_list:
            for t in target:
                if t not in table:
                    table[t] = 1
                else:
                    table[t] += 1
        ### Step 2. Top k list for encode map
        max_idx = min(max_idx-3,len(table))
        all_tokens = [k for k,v in sorted(table.items(), key = itemgetter(1), reverse = True)][:max_idx]
        table = {'<sos>':0,'<eos>':1}
        if mode == "word": table['<unk>']=2
        for tok in all_tokens:
            table[tok] = len(table)
    ### Step 3. Encode
    output_list = []
    for target in input_list:
        tmp = [0]
        for t in target:
            if t in table:
                tmp.append(table[t])
            else:
                if mode == "word":
                    tmp.append(2)
                else:
                    tmp.append(table['<unk>'])
                    # raise ValueError('OOV error: '+t)
        tmp.append(1)
        output_list.append(tmp)
    return output_list,table


# Feature Padding Function 
# Parameters
#     - x          : list, list of np.array
#     - pad_len    : int, length to pad (0 for max_len in x)      
# Return
#     - new_x      : np.array with shape (len(x),pad_len,dim of feature)
def zero_padding(x,pad_len):
    features = x[0].shape[-1]
    if pad_len is 0: pad_len = max([len(v) for v in x])
    new_x = np.zeros((len(x),pad_len,features))
    for idx,ins in enumerate(x):
        new_x[idx,:min(len(ins),pad_len),:] = ins[:min(len(ins),pad_len),:]
    return new_x

# Target Padding Function 
# Parameters
#     - y          : list, list of int
#     - max_len    : int, max length of output (0 for max_len in y)     
# Return
#     - new_y      : np.array with shape (len(y),max_len)
def target_padding(y,max_len):
    if max_len is 0: max_len = max([len(v) for v in y])
    new_y = np.zeros((len(y),max_len),dtype=int)
    for idx,label_seq in enumerate(y):
        new_y[idx,:len(label_seq)] = np.array(label_seq)
    return new_y
