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
hidden_size = 256
num_layers = 2

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, dim, layer, rnn_cell, dropout):
        super(LSTMClassifier, self).__init__()
        assert "Cell" in rnn_cell,'Please use Recurrent Cell instead of layer in decoder'
        # Manually forward through Cells if using RNNCell family
        self.layer = layer
        self.dim = dim
        self.dropout = nn.Dropout(p=dropout)
        
        self.label = nn.Linear(dim, output_size)
        self.softmax = nn.LogSoftmax()

        self.layer0 = getattr(nn,rnn_cell)(input_dim,dim)
        for i in range(1,layer):
            setattr(self,'layer'+str(i), getattr(nn,rnn_cell)(dim,dim))
        
        self.state_list = []
        self.cell_list = []
        
        #self.layer = RNNLayer(input_dim,dim, 1, rnn_cell=rnn_cell, layers=layer,
        #                                           dropout_rate=dropout, bidir=False)
    def init_rnn(self,context):
        self.state_list = [torch.zeros(context.shape[0],self.dim).to(context.device)]*self.layer
        self.cell_list = [torch.zeros(context.shape[0],self.dim).to(context.device)]*self.layer

    @property
    def hidden_state(self):
        return [s.clone().detach().cpu() for s in self.state_list], [c.clone().detach().cpu() for c in self.cell_list]

    @hidden_state.setter
    def hidden_state(self, state): # state is a tuple of two list
        device = self.state_list[0].device
        self.state_list = [s.to(device) for s in state[0]]
        self.cell_list = [c.to(device) for c in state[1]]
    
    def forward(self, input_context):
        input_context = input_context.permute(0, 2, 1)
        self.state_list[0],self.cell_list[0] = self.layer0(self.dropout(input_context),(self.state_list[0],self.cell_list[0]))
        for l in range(1,self.layer):
            self.state_list[l],self.cell_list[l] = getattr(self,'layer'+str(l))(self.state_list[l-1],(self.dropout(self.state_list[l]),self.cell_list[l]))
        
        #final_output = self.dropout_layer(final_hidden_state[-1])
        final_output = self.label(self.state_list[-1])
        final_output = self.softmax(final_output)
        return final_output


class LSTMClassifier_old(nn.Module):

	def __init__(self, input_dim):
		super(LSTMClassifier_old, self).__init__()
		
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
		
		###self.device = device

		#self.batch_size = batch_size
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.input_dim = input_dim
		#self.embedding_length = embedding_length
		
		#self.word_embeddings = nn.Embedding(vocab_size, embedding_length)# Initializing the look-up table.
		#if model_level == "word":
	    #   self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False) # Assigning the look-up table to the pre-trained GloVe word embedding.
		
		self.lstm = nn.LSTM(self.input_dim, hidden_size, dropout=.3, num_layers=num_layers)
		self.label = nn.Linear(hidden_size, output_size)
		self.softmax = nn.Softmax()

		self.dropout_layer = nn.Dropout(p=0.3)


    
    
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
		hidden = (weight.new(num_layers, batch_size, self.hidden_size).zero_(), weight.new(num_layers, batch_size, self.hidden_size).zero_())
		#hidden = (weight.new(num_layers, batch_size, self.hidden_size).zero_().to(self.device), weight.new(num_layers, batch_size, self.hidden_size).zero_().to(self.device))

		output, (final_hidden_state, final_cell_state) = self.lstm(input, hidden)
		#final_output = self.label(final_hidden_state[-1]) # final_hidden_state.size() = (1, batch_size, hidden_size) & final_output.size() = (batch_size, output_size)
		
		final_output = self.dropout_layer(final_hidden_state[-1])
		logits = self.label(final_output)
		probs = self.softmax(logits)

		return logits, probs
