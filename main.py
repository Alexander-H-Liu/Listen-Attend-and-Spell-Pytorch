#!/usr/bin/env python
# coding: utf-8
import yaml
import torch
import random
import argparse
import numpy as np
import os
import logging

# Make cudnn CTC deterministic
torch.backends.cudnn.deterministic = True

# Arguments
parser = argparse.ArgumentParser(description='Training E2E asr.')
parser.add_argument('--config', type=str, help='Path to experiment config.')
parser.add_argument('--name', default=None, type=str, help='Name for logging.')
parser.add_argument('--logdir', default='log/', type=str, help='Logging path.', required=False)
parser.add_argument('--ckpdir', default='result/', type=str, help='Checkpoint/Result path.', required=False)
parser.add_argument('--load', default=None, type=str, help='Load pre-trained model', required=False)
parser.add_argument('--seed', default=0, type=int, help='Random seed for reproducable results.', required=False)
parser.add_argument('--njobs', default=1, type=int, help='Number of threads for decoding.', required=False)
parser.add_argument('--gpu_no', type=int, help='which gpu to use (in case od cpu not effective)', required=True)
parser.add_argument('--cpu', action='store_true', help='Disable GPU training.')
parser.add_argument('--test', action='store_true', help='Test the model.')
parser.add_argument('--valid', action='store_true', help='valid the model.')
parser.add_argument('--no-msg', action='store_true', help='Hide all messages.')
parser.add_argument('--rnnlm', action='store_true', help='Option for training RNNLM.')
parser.add_argument('--asr_model_file', default=None, type=str, help='asr model file (pre-trained model or for valdiation/prediction)')
parser.add_argument('--acoustic_classifier_model_file', default=None, type=str, help='acoustic classifier model file (pre-trained model or for valdiation/prediction)')

paras = parser.parse_args()
setattr(paras,'gpu',not paras.cpu)
setattr(paras,'verbose',not paras.no_msg)
config = yaml.load(open(paras.config,'r'))
print("config file: ", paras.config)
random.seed(paras.seed)
np.random.seed(paras.seed)
torch.manual_seed(paras.seed)
if torch.cuda.is_available(): 
    torch.cuda.manual_seed_all(paras.seed)
    print("GPU NO: " + str(paras.gpu_no))
    os.environ["CUDA_VISIBLE_DEVICES"]=str(paras.gpu_no)
    torch.cuda.set_device(paras.gpu_no)  # TODO : set this from cmd/config

if not paras.rnnlm:
    if not paras.test and not paras.valid:
        # Train ASR
        from src.solver import Trainer as Solver
    elif paras.test:
        # Test ASR
        from src.solver import Tester as Solver
    elif paras.valid:
        # Valdiation (prediction + some acuracy comutation using labels)
        from src.solver import Validator as Solver
else:
    # Train RNNLM
    from src.solver import RNNLM_Trainer as Solver

solver = Solver(config,paras)
solver.load_data()
solver.set_model()
solver.exec()
