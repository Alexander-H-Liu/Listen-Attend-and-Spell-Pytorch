#!/usr/bin/env python
# coding: utf-8
import yaml
import torch
import random
import argparse
import numpy as np

# Make cudnn CTC deterministic
torch.backends.cudnn.deterministic = True

# Arguments
parser = argparse.ArgumentParser(description='Training E2E asr.')
parser.add_argument('--config', type=str, help='Path to experiment config.')
parser.add_argument('--name', default=None, type=str, help='Name for logging.')
parser.add_argument('--logdir', default='log/', type=str, help='Logging path.', required=False)
parser.add_argument('--ckpdir', default='checkpoint/', type=str, help='Checkpoint path.', required=False)
parser.add_argument('--load', default=None, type=str, help='Load pre-trained model', required=False)
parser.add_argument('--seed', default=0, type=int, help='Random seed for reproducable results.', required=False)
parser.add_argument('--cpu', action='store_true', help='Disable GPU training.')
parser.add_argument('--test', action='store_true', help='Test the model.')
parser.add_argument('--rnnlm', action='store_true', help='Option for training RNNLM.')
paras = parser.parse_args()
setattr(paras,'gpu',not paras.cpu)
config = yaml.load(open(paras.config,'r'))

random.seed(paras.seed)
np.random.seed(paras.seed)
torch.manual_seed(paras.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(paras.seed)

if not paras.rnnlm:
    if not paras.test:
        # Train ASR
        from src.solver import Trainer
        solver = Trainer(config,paras)
    else:
        # Test ASR
        from src.solver import Tester
        solver = Tester(config,paras)
else:
    # Train RNNLM
    from src.solver import RNNLM_Trainer
    solver = RNNLM_Trainer(config,paras)

solver.load_data()
solver.set_model()
solver.exec()
