import yaml
import os
from util.librispeech_dataset import create_dataloader
from util.functions import log_parser,batch_iterator, collapse_phn
from model.las_model import Listener,Speller
import numpy as np
from torch.autograd import Variable
import torch
import time
from tensorboardX import SummaryWriter

import argparse


parser = argparse.ArgumentParser(description='Training script for LAS on Librispeech .')

parser.add_argument('config_path', metavar='config_path', type=str,
                     help='Path to config file for training.')

paras = parser.parse_args()

config_path = paras.config_path

# Load config file for experiment
print('Loading configure file at',config_path)
conf = yaml.load(open(config_path,'r'))

# Parameters loading
print()
print('Experiment :',conf['meta_variable']['experiment_name'])
total_steps = conf['training_parameter']['total_steps']

listener_model_path = conf['meta_variable']['checkpoint_dir']+conf['meta_variable']['experiment_name']+'.listener'
speller_model_path = conf['meta_variable']['checkpoint_dir']+conf['meta_variable']['experiment_name']+'.speller'
verbose_step = conf['training_parameter']['verbose_step']
valid_step = conf['training_parameter']['valid_step']
tf_rate_upperbound = conf['training_parameter']['tf_rate_upperbound']
tf_rate_lowerbound = conf['training_parameter']['tf_rate_lowerbound']
tf_decay_step = conf['training_parameter']['tf_decay_step']
seed = conf['training_parameter']['seed']

# Fix random seed
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Load preprocessed LibriSpeech Dataset

train_set = create_dataloader(conf['meta_variable']['data_path']+'/train.csv', 
                              **conf['model_parameter'], **conf['training_parameter'], shuffle=True,training=True)
valid_set = create_dataloader(conf['meta_variable']['data_path']+'/dev.csv',
                              **conf['model_parameter'], **conf['training_parameter'], shuffle=False,drop_last=True)

idx2char = {}
with open(conf['meta_variable']['data_path']+'/idx2chap.csv','r') as f:
    for line in f:
        if 'idx' in line:continue
        idx2char[int(line.split(',')[0])] = line[:-1].split(',')[1]

# Load pre-trained model if needed
if conf['training_parameter']['use_pretrained']:
    global_step = conf['training_parameter']['pretrained_step']
    listener = torch.load(conf['training_parameter']['pretrained_listener_path'])
    speller = torch.load(conf['training_parameter']['pretrained_speller_path'])
else:
    global_step = 0
    listener = Listener(**conf['model_parameter'])
    speller = Speller(**conf['model_parameter'])
optimizer = torch.optim.Adam([{'params':listener.parameters()}, {'params':speller.parameters()}], 
                              lr=conf['training_parameter']['learning_rate'],
                              betas=(0.5, 0.999),weight_decay=1e-5,amsgrad=True)

best_ler = 1.0
record_gt_text = False
log_writer = SummaryWriter(conf['meta_variable']['training_log_dir']+conf['meta_variable']['experiment_name'])

# Training
print('Training starts...',flush=True)
while global_step<total_steps:

    # Teacher forcing rate linearly decay
    tf_rate = tf_rate_upperbound - (tf_rate_upperbound-tf_rate_lowerbound)*min((float(global_step)/tf_decay_step),1)
    
    
    # Training
    for batch_data,batch_label in train_set:
        print('Current step :',global_step,end='\r',flush=True)
        
        batch_loss, batch_ler = batch_iterator(batch_data, batch_label, listener, speller, optimizer, tf_rate,
                                               is_training=True, data='libri', **conf['model_parameter'])
        global_step += 1

        if (global_step) % verbose_step == 0:
            log_writer.add_scalars('loss',{'train':batch_loss}, global_step)
            log_writer.add_scalars('cer',{'train':np.array([np.array(batch_ler).mean()])}, global_step)
        
        if global_step % valid_step == 0:
            break

    
    # Validation
    val_loss = []
    val_ler = []
    
    for batch_data,batch_label in valid_set:
        batch_loss, batch_ler = batch_iterator(batch_data, batch_label, listener, speller, optimizer, 
                                               tf_rate, is_training=False, data='libri', **conf['model_parameter'])
        val_loss.append(batch_loss)
        val_ler.extend(batch_ler)
    
    
    val_loss = np.array([sum(val_loss)/len(val_loss)])
    val_ler = np.array([sum(val_ler)/len(val_ler)])
    log_writer.add_scalars('loss',{'dev':val_loss}, global_step)
    log_writer.add_scalars('cer',{'dev':val_ler}, global_step)

    
    # Generate Example
    if conf['model_parameter']['bucketing']:
        feature = listener(Variable(batch_data.float()).squeeze().cuda())
    else:
        feature = listener(Variable(batch_data.float()).squeeze().cuda())
    pred_seq, attention_score = speller(feature)
    
    pred_seq = [char.cpu() for char in pred_seq]
    for t in range(len(attention_score)):
        for h in range(len(attention_score[t])):
            attention_score[t][h] = attention_score[t][h].cpu()
    del feature
    
    
    pd = {i:'' for i in range(conf['training_parameter']['batch_size'])}
    for t,char in enumerate(pred_seq):
        for idx,i in enumerate(torch.max(char,dim=-1)[1]):
            if int(i) == 0: continue
            if int(i) == 1: break
            pd[idx] += idx2char[int(i)]
    pd = [pd[i] for i in range(conf['training_parameter']['batch_size'])]

    gt = []
    for line in (torch.max(batch_label.squeeze(),dim=-1)[1]).numpy():
        tmp = ''
        for idx in line:
            if idx == 0: continue
            if idx == 1: break
            tmp += idx2char[idx]
        gt.append(tmp)
    
    for idx,(p,g) in enumerate(zip(pd,gt)):
        log_writer.add_text('pred_'+str(idx), p, global_step)
        if not record_gt_text:
            log_writer.add_text('test_'+str(idx), g, global_step)
    
    if not record_gt_text:
        record_gt_text = True
        
    att_map = {i:[] for i in range(conf['training_parameter']['batch_size'])}
    num_head = len(attention_score[0])
    for i in range(conf['training_parameter']['batch_size']):
        for j in range(num_head):
            att_map[i].append([])
    for t,head_score in enumerate(attention_score):
        for h,att_score in enumerate(head_score):
            for idx,att in enumerate(att_score.data.numpy()):
                att_map[idx][h].append(att)
    for i in range(conf['training_parameter']['batch_size']):
        for j in range(num_head):
            m = np.repeat(np.expand_dims(np.array(att_map[i][j]),0),3,axis=0)
            log_writer.add_image('attention_'+str(i)+'_head_'+str(j),
                                 torch.FloatTensor(m[:,:len(pd[i]),:]), global_step)
    
    # Checkpoint
    if best_ler >= sum(val_ler)/len(val_ler):
        best_ler = sum(val_ler)/len(val_ler)
        print('Reached best CER',best_ler,'at step',global_step,',checkpoint saved.')
        torch.save(listener, listener_model_path)
        torch.save(speller, speller_model_path)
