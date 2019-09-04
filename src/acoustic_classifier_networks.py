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

	def __init__(self, example, model_para, asr_model_para):
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

		if model_para["input"] == "VGG":
			_,_, enc_out_dim = self.check_dim(example)

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


	def check_dim(self, example_input):
		d = example_input.shape[-1]
		if d%13 == 0:
			# MFCC feature
			return int(d/13),13,(13//4)*128
		elif d%40 == 0:
			# Fbank feature
			return int(d/40),40,(40//4)*128
		else:
			raise ValueError('Acoustic feature dimension for VGG should be 13/26/39(MFCC) or 40/80/120(Fbank) but got '+d)

    
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
		output, (final_hidden_state, final_cell_state) = self.lstm(input, hidden)
		final_output = self.dropout_layer(final_hidden_state[-1])
		logits = self.label(final_output)
		probs = self.softmax(logits)

		return logits, probs



class AttentionModel(torch.nn.Module):
	def __init__(self, example, model_para, asr_model_para):
		super(AttentionModel, self).__init__()
		
		"""
		Arguments
		---------
		batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
		output_size : 2 = (pos, neg)
		hidden_sie : Size of the hidden_state of the LSTM
		vocab_size : Size of the vocabulary containing unique words
		embedding_length : Embeddding dimension of GloVe word embeddings
		weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 
		
		--------
		
		"""
		enc_out_dim = int(asr_model_para['encoder']['dim'].split('_')[-1])\
                      *max(1,2*('Bi' in asr_model_para['encoder']['enc_type']))\
                      *max(1,int(asr_model_para['encoder']['sample_rate'].split('_')[-1])\
                           *('concat'== asr_model_para['encoder']['sample_style']))

		if model_para["input"] == "VGG":
			_,_, enc_out_dim = self.check_dim(example)

		###self.device = device

		#self.batch_size = batch_size
		self.output_size = output_size
		self.input_dim = enc_out_dim
		self.num_layers = model_para["lstm_num_layers"]
		rnn_dropout = model_para["lstm_rnn_dropout"]
		self.hidden_size = model_para["lstm_hidden_size"]
		last_dropout = model_para["lstm_final_dropout"]

		#self.batch_size = batch_size
		#self.output_size = output_size
		#self.hidden_size = hidden_size
		#self.vocab_size = vocab_size
		#self.embedding_length = embedding_length
		
		#self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
		#if modeling_level == "word":
		#	self.word_embeddings.weights = nn.Parameter(weights, requires_grad=False)
		self.lstm = nn.LSTM(self.input_dim, self.hidden_size, dropout=rnn_dropout, num_layers=self.num_layers)
		self.label = nn.Linear(self.hidden_size, self.output_size)
		self.dropout_layer = nn.Dropout(p=last_dropout)
		#self.attn_fc_layer = nn.Linear()
		self.softmax = nn.Softmax()

	def check_dim(self, example_input):
		d = example_input.shape[-1]
		if d%13 == 0:
			# MFCC feature
			return int(d/13),13,(13//4)*128
		elif d%40 == 0:
			# Fbank feature
			return int(d/40),40,(40//4)*128
		else:
			raise ValueError('Acoustic feature dimension for VGG should be 13/26/39(MFCC) or 40/80/120(Fbank) but got '+d)
	
	def attention_net(self, lstm_output, final_state):

		""" 
		Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
		between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.
		
		Arguments
		---------
		
		lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
		final_state : Final time-step hidden state (h_n) of the LSTM
		
		---------
		
		Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
				  new hidden state.
				  
		Tensor Size :
					hidden.size() = (batch_size, hidden_size)
					attn_weights.size() = (batch_size, num_seq)
					soft_attn_weights.size() = (batch_size, num_seq)
					new_hidden_state.size() = (batch_size, hidden_size)
					  
		"""
		
		hidden = final_state.squeeze(0)
		attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
		soft_attn_weights = F.softmax(attn_weights, 1)
		new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
		
		return new_hidden_state
	
	def forward(self, input_sentences, batch_size=None):
	
		""" 
		Parameters
		----------
		input_sentence: input_sentence of shape = (batch_size, num_sequences)
		batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)
		
		Returns
		-------
		Output of the linear layer containing logits for pos & neg class which receives its input as the new_hidden_state which is basically the output of the Attention network.
		final_output.shape = (batch_size, output_size)
		
		"""
		
		#input = self.word_embeddings(input_sentences)
		input = input_sentences.permute(1, 0, 2)
		"""
		if batch_size is None:
			h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
			c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
		else:
			h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
			c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
		"""
		weight = next(self.parameters()).data
		hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_(), weight.new(self.num_layers, batch_size, self.hidden_size).zero_())
			
		output, (final_hidden_state, final_cell_state) = self.lstm(input, hidden) # final_hidden_state.size() = (1, batch_size, hidden_size) 
		output = output.permute(1, 0, 2) # output.size() = (batch_size, num_seq, hidden_size)
		
		attn_output = self.attention_net(output, final_hidden_state)
		final_output = self.dropout_layer(attn_output[-1])
		
		logits = self.label(final_output)
		probs = self.softmax(logits)
		
		return logits, probs


class SelfAttention(nn.Module):
	def __init__(self, example, model_para, asr_model_para):
		super(SelfAttention, self).__init__()

		"""
		Arguments
		---------
		batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
		output_size : 2 = (pos, neg)
		hidden_sie : Size of the hidden_state of the LSTM
		vocab_size : Size of the vocabulary containing unique words
		embedding_length : Embeddding dimension of GloVe word embeddings
		weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 
		
		--------
		
		"""
		enc_out_dim = int(asr_model_para['encoder']['dim'].split('_')[-1])\
                      *max(1,2*('Bi' in asr_model_para['encoder']['enc_type']))\
                      *max(1,int(asr_model_para['encoder']['sample_rate'].split('_')[-1])\
                           *('concat'== asr_model_para['encoder']['sample_style']))

		if model_para["input"] == "VGG":
			_,_, enc_out_dim = self.check_dim(example)

		###self.device = device

		#self.batch_size = batch_size
		self.output_size = output_size
		self.input_dim = enc_out_dim
		self.num_layers = model_para["lstm_num_layers"]
		rnn_dropout = model_para["lstm_rnn_dropout"]
		self.hidden_size = model_para["lstm_hidden_size"]
		last_dropout = model_para["lstm_final_dropout"]

		#self.batch_size = batch_size
		#self.output_size = output_size
		#self.hidden_size = hidden_size
		#self.vocab_size = vocab_size
		#self.embedding_length = embedding_length
		#self.weights = weights

		s#elf.word_embeddings = nn.Embedding(vocab_size, embedding_length)
		#if model_level == "word":
		#	self.word_embeddings.weights = nn.Parameter(weights, requires_grad=False)
		#self.dropout = 0.8
		self.bilstm = nn.LSTM(self.input_dim, self.hidden_size, dropout=rnn_dropout, bidirectional=True, num_layers=self.num_layers)
		# We will use da = 350, r = 30 & penalization_coeff = 1 as per given in the self-attention original ICLR paper
		self.W_s1 = nn.Linear(2*self.hidden_size, 350)
		self.W_s2 = nn.Linear(350, 30)
		self.fc_layer = nn.Linear(30*2*self.hidden_size, 2000)
		self.label = nn.Linear(2000, self.output_size)
		self.dropout_layer = nn.Dropout(p=last_dropout)
		self.softmax = nn.Softmax()

	def attention_net(self, lstm_output):

		"""
		Now we will use self attention mechanism to produce a matrix embedding of the input sentence in which every row represents an
		encoding of the inout sentence but giving an attention to a specific part of the sentence. We will use 30 such embedding of 
		the input sentence and then finally we will concatenate all the 30 sentence embedding vectors and connect it to a fully 
		connected layer of size 2000 which will be connected to the output layer of size 2 returning logits for our two classes i.e., 
		pos & neg.

		Arguments
		---------

		lstm_output = A tensor containing hidden states corresponding to each time step of the LSTM network.
		---------

		Returns : Final Attention weight matrix for all the 30 different sentence embedding in which each of 30 embeddings give
				  attention to different parts of the input sentence.

		Tensor size : lstm_output.size() = (batch_size, num_seq, 2*hidden_size)
					  attn_weight_matrix.size() = (batch_size, 30, num_seq)

		"""
		attn_weight_matrix = self.W_s2(F.tanh(self.W_s1(lstm_output)))
		attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
		attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

		return attn_weight_matrix

	def forward(self, input_sentences, batch_size):

		""" 
		Parameters
		----------
		input_sentence: input_sentence of shape = (batch_size, num_sequences)
		batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)
		
		Returns
		-------
		Output of the linear layer containing logits for pos & neg class.
		
		"""

		input = input_sentences.permute(1, 0, 2)
		"""
		if batch_size is None:
			h_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size).cuda())
			c_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size).cuda())
		else:
			h_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).cuda())
			c_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).cuda())
		"""
		weight = next(self.parameters()).data
		hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_(), weight.new(self.num_layers, batch_size, self.hidden_size).zero_())

		output, (h_n, c_n) = self.bilstm(input, hidden)
		output = output.permute(1, 0, 2)
		# output.size() = (batch_size, num_seq, 2*hidden_size)
		# h_n.size() = (1, batch_size, hidden_size)
		# c_n.size() = (1, batch_size, hidden_size)
		attn_weight_matrix = self.attention_net(output)
		# attn_weight_matrix.size() = (batch_size, r, num_seq)
		# output.size() = (batch_size, num_seq, 2*hidden_size)
		hidden_matrix = torch.bmm(attn_weight_matrix, output)
		# hidden_matrix.size() = (batch_size, r, 2*hidden_size)
		# Let's now concatenate the hidden_matrix and connect it to the fully connected layer.
		fc_out = self.fc_layer(hidden_matrix.view(-1, hidden_matrix.size()[1]*hidden_matrix.size()[2]))
		drop_output = self.dropout_layer(fc_out)
		logits = self.label(drop_output)
		# logits.size() = (batch_size, output_size)
		probs = self.softmax(logits)

		return logits, probs