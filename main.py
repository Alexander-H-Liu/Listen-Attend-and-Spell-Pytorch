#!/usr/bin/env python
# coding: utf-8
import yaml
import torch
import random
import argparse
import numpy as np

from src.trainer import Trainer

# Make cudnn CTC deterministic
torch.backends.cudnn.deterministic = True

# In[ ]:


# Arguments
parser = argparse.ArgumentParser(description='Training E2E asr.')
parser.add_argument('--config', type=str, help='Path to experiment config.')
parser.add_argument('--name', default=None, type=str, help='Name for logging.')
parser.add_argument('--logdir', default='log/', type=str, help='Logging path.', required=False)
parser.add_argument('--ckpdir', default='checkpoint/', type=str, help='Checkpoint path.', required=False)
parser.add_argument('--load', default=None, type=str, help='Load pre-trained model', required=False)
parser.add_argument('--seed', default=531, type=int, help='Random seed for reproducable results.', required=False)
parser.add_argument('--gpu', action='store_true', help='Enable GPU training.')
parser.add_argument('--dev_set', default='test', type=str, help='Specify Develop set.', required=False)
parser.add_argument('--test', action='store_true', help='Test the model.')
paras = parser.parse_args()

config = yaml.load(open(paras.config,'r'))

random.seed(paras.seed)
np.random.seed(paras.seed)
torch.manual_seed(paras.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(paras.seed)


# Setup trainer
trainer = Trainer(config,paras)
trainer.load_data()

trainer.set_model()

if not paras.test:
    trainer.train()
else:
    trainer.inference()
