# acoustic_classifier_networks.py 
# Aug 2019 , Amir Harati
"""
    Acoustic classifier networks 
"""

import os
import torch 
import random
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
import math
import torch
from torch.autograd import Variable
from torch.nn import functional as F

output_size = 2 




class LSTMClassifier(nn.Module):

	def __init__(self, model_para, asr_model_para):
		super(LSTMClassifier, self).__init__()
		
		"""
		Arguments
		---------
		batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
		output_size : 2 = (pos, neg)
		hidden_sie : Size of the hidden_state of the LSTM
		vocab_size : Size of the vocabulary containing unique words
		embedding_length : Embeddding dimension of GloVe word embeddings
		weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 
		
		"""
		enc_out_dim = int(asr_model_para['encoder']['dim'].split('_')[-1])\
                      *max(1,2*('Bi' in asr_model_para['encoder']['enc_type']))\
                      *max(1,int(asr_model_para['encoder']['sample_rate'].split('_')[-1])\
                           *('concat'== asr_model_para['encoder']['sample_style']))

		###self.device = device

		#self.batch_size = batch_size
		self.output_size = output_size
		self.input_dim = enc_out_dim
		self.num_layers = model_para["lstm_num_layers"]
		rnn_dropout = model_para["lstm_rnn_dropout"]
		self.hidden_size = model_para["lstm_hidden_size"]
		last_dropout = model_para["lstm_final_dropout"]

		#self.embedding_length = embedding_length
		
		#self.word_embeddings = nn.Embedding(vocab_size, embedding_length)# Initializing the look-up table.
		#if model_level == "word":
	    #   self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False) # Assigning the look-up table to the pre-trained GloVe word embedding.
		
		self.lstm = nn.LSTM(self.input_dim, self.hidden_size, dropout=rnn_dropout, num_layers=self.num_layers)
		self.label = nn.Linear(self.hidden_size, output_size)
		self.softmax = nn.Softmax()

		self.dropout_layer = nn.Dropout(p=last_dropout)


    
    
	def forward(self, input_sentence, batch_size):
	
		""" 
		Parameters
		----------
		input_sentence: input_sentence of shape = (batch_size, num_sequences)
		batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)
		
		Returns
		-------
		Output of the linear layer containing logits for positive & negative class which receives its input as the final_hidden_state of the LSTM
		final_output.shape = (batch_size, output_size)
		
		"""
		
		''' Here we will map all the indexes present in the input sequence to the corresponding word vector using our pre-trained word_embedddins.'''
		#input = self.word_embeddings(input_sentence) # embedded input of shape = (batch_size, num_sequences,  embedding_length)
		input = input_sentence.permute(1, 0, 2) # input.size() = (num_sequences, batch_size, embedding_length)
		"""
		if batch_size is None:
			h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda()) # Initial hidden state of the LSTM
			c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda()) # Initial cell state of the LSTM
		else:
			h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
			c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
		"""
		#if batch_size is None:
	    #   batch_size =  self.batch_size
		weight = next(self.parameters()).data
		hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_(), weight.new(self.num_layers, batch_size, self.hidden_size).zero_())
		#hidden = (weight.new(num_layers, batch_size, self.hidden_size).zero_().to(self.device), weight.new(num_layers, batch_size, self.hidden_size).zero_().to(self.device))

		output, (final_hidden_state, final_cell_state) = self.lstm(input, hidden)
		#final_output = self.label(final_hidden_state[-1]) # final_hidden_state.size() = (1, batch_size, hidden_size) & final_output.size() = (batch_size, output_size)
		
		final_output = self.dropout_layer(final_hidden_state[-1])
		logits = self.label(final_output)
		probs = self.softmax(logits)

		return logits, probs
