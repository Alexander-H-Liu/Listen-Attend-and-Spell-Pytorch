import yaml
from util.timit_dataset import load_dataset,create_dataloader
from model.las_model import Listener,Speller
from util.functions import batch_iterator
import numpy as np
from torch.autograd import Variable
import torch
import sys
import time

# Load config file for experiment
try:
    config_path = sys.argv[1]
    conf = yaml.load(open(config_path,'r'))
except:
    print('Usage: python3 run_exp.py <config file path>')

# Parameters loading
torch.manual_seed(conf['training_parameter']['seed'])
num_epochs = conf['training_parameter']['num_epochs']
use_pretrained = conf['training_parameter']['use_pretrained']
training_msg = 'epoch_{:2d}_step_{:3d}_TrLoss_{:.4f}_TrWER_{:.2f}'
epoch_end_msg = 'epoch_{:2d}_TrLoss_{:.4f}_TrWER_{:.2f}_TtLoss_{:.4f}_TtWER_{:.2f}_time_{:.2f}'
verbose_step = conf['training_parameter']['verbose_step']
tf_rate_upperbound = conf['training_parameter']['tf_rate_upperbound']
tf_rate_lowerbound = conf['training_parameter']['tf_rate_lowerbound']

# Load preprocessed TIMIT Dataset ( using testing set directly here, replace them with validation set your self)
# X : Padding to shape [num of sample, max_timestep, feature_dim]
# Y : Squeeze repeated label and apply one-hot encoding (preserve 0 for <sos> and 1 for <eos>)
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(**conf['meta_variable'])
train_set = create_dataloader(X_train, y_train, **conf['model_parameter'], **conf['training_parameter'], shuffle=True)
valid_set = create_dataloader(X_val, y_val, **conf['model_parameter'], **conf['training_parameter'], shuffle=False)
test_set = create_dataloader(X_test, y_test, **conf['model_parameter'], **conf['training_parameter'], shuffle=False)

# Construct LAS Model or load pretrained LAS model
if not use_pretrained:
    traing_log = open(conf['meta_variable']['training_log_dir']+conf['meta_variable']['experiment_name']+'.log','w')
    listener = Listener(**conf['model_parameter'])
    speller = Speller(**conf['model_parameter'])
else:
    traing_log = open(conf['meta_variable']['training_log_dir']+conf['meta_variable']['experiment_name']+'.log','a')
    listener = torch.load(conf['training_parameter']['pretrained_listener_path'])
    speller = torch.load(conf['training_parameter']['pretrained_speller_path'])
optimizer = torch.optim.Adam([{'params':listener.parameters()}, {'params':speller.parameters()}],
                             lr=conf['training_parameter']['learning_rate'])
listener_model_path = conf['meta_variable']['checkpoint_dir']+conf['meta_variable']['experiment_name']+'.listener'
speller_model_path = conf['meta_variable']['checkpoint_dir']+conf['meta_variable']['experiment_name']+'.speller'

# save checkpoint with the best ler
best_ler = 1.0

for epoch in range(num_epochs):
    epoch_head = time.time()
    tr_loss = 0.0
    tr_ler = []
    tt_loss = 0.0
    tt_ler = []

    # Teacher forcing rate linearly decay
    tf_rate = tf_rate_upperbound - (tf_rate_upperbound-tf_rate_lowerbound)*(epoch/num_epochs)
    
    # Training
    for batch_index,(batch_data,batch_label) in enumerate(train_set):
        batch_loss, batch_ler = batch_iterator(batch_data, batch_label, listener, speller, optimizer, 
                                               tf_rate, is_training=True, **conf['model_parameter'])
        tr_loss += batch_loss
        tr_ler.extend(batch_ler)
        if (batch_index+1) % verbose_step == 0:
            print(training_msg.format(epoch+1,batch_index+1,tr_loss[0]/(batch_index+1),sum(tr_ler)/len(tr_ler)),end='\r',flush=True)
    training_time = float(time.time()-epoch_head)
    
    # Testing
    for _,(batch_data,batch_label) in enumerate(test_set):
        batch_loss, batch_ler = batch_iterator(batch_data, batch_label, listener, speller, optimizer, 
                                               tf_rate, is_training=False, **conf['model_parameter'])
        tt_loss += batch_loss
        tt_ler.extend(batch_ler)

    # Logger
    print(epoch_end_msg.format(epoch+1,tr_loss[0]/(batch_index+1),sum(tr_ler)/len(tr_ler),
                               tt_loss[0]/len(test_set),sum(tt_ler)/len(tt_ler),training_time),flush=True)
    print(epoch_end_msg.format(epoch+1,tr_loss[0]/(batch_index+1),sum(tr_ler)/len(tr_ler),
                               tt_loss[0]/len(test_set),sum(tt_ler)/len(tt_ler),training_time),flush=True,file=traing_log)

    # Checkpoint
    if best_ler >= sum(tt_ler)/len(tt_ler):
        best_ler = sum(tt_ler)/len(tt_ler)
        torch.save(listener, listener_model_path)
        torch.save(speller, speller_model_path)
