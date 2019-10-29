import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from src.util import init_weights, init_gate
from src.module import VGGExtractor, RNNLayer, ScaleDotAttention, LocationAwareAttention


class ASR(nn.Module):
    ''' ASR model, including Encoder/Decoder(s)'''

    def __init__(self, input_size, vocab_size, ctc_weight, encoder, attention, decoder, emb_drop=0.0):
        super(ASR, self).__init__()

        # Setup
        assert 0 <= ctc_weight <= 1
        self.vocab_size = vocab_size
        self.ctc_weight = ctc_weight
        self.enable_ctc = ctc_weight > 0
        self.enable_att = ctc_weight != 1
        self.lm = None

        # Modules
        self.encoder = Encoder(input_size, **encoder)
        if self.enable_ctc:
            self.ctc_layer = nn.Linear(self.encoder.out_dim, vocab_size)
        if self.enable_att:
            self.dec_dim = decoder['dim']
            self.pre_embed = nn.Embedding(vocab_size, self.dec_dim)
            self.embed_drop = nn.Dropout(emb_drop)
            self.decoder = Decoder(
                self.encoder.out_dim+self.dec_dim, vocab_size, **decoder)
            query_dim = self.dec_dim*self.decoder.layer
            self.attention = Attention(
                self.encoder.out_dim, query_dim, **attention)

        # Init
        self.apply(init_weights)
        for l in range(self.decoder.layer):
            bias = getattr(self.decoder.layers, 'bias_ih_l{}'.format(l))
            bias = init_gate(bias)

    def set_state(self, prev_state, prev_attn):
        ''' Setting up all memory states for beam decoding'''
        self.decoder.set_state(prev_state)
        self.attention.set_mem(prev_attn)

    def create_msg(self):
        # Messages for user
        msg = []
        msg.append('Model spec.| Encoder\'s downsampling rate of time axis is {}.'.format(
            self.encoder.sample_rate))
        if self.encoder.vgg:
            msg.append(
                '           | VCC Extractor w/ time downsampling rate = 4 in encoder enabled.')
        if self.enable_ctc:
            msg.append('           | CTC training on encoder enabled ( lambda = {}).'.format(
                self.ctc_weight))
        if self.enable_att:
            msg.append('           | {} attention decoder enabled ( lambda = {}).'.format(
                self.attention.mode, 1-self.ctc_weight))
        return msg

    def forward(self, audio_feature, feature_len, decode_step, tf_rate=0.0, teacher=None,
                emb_decoder=None, get_dec_state=False):
        '''
        Arguments
            audio_feature - [BxTxD] Acoustic feature with shape 
            feature_len   - [B]     Length of each sample in a batch
            decode_step   - [int]   The maximum number of attention decoder steps 
            tf_rate       - [0,1]   The probability to perform teacher forcing for each step
            teacher       - [BxLxD] Ground truth for teacher forcing with sentence length L
            emb_decoder   - [obj]   Introduces the word embedding decoder, different behavior for training/inference
                                    At training stage, this ONLY affects self-sampling (output remains the same)
                                    At inference stage, this affects output to become log prob. with distribution fusion
            get_dec_state - [bool]  If true, return decoder state [BxLxD] for other purpose
        '''
        # Init
        bs = audio_feature.shape[0]
        ctc_output, att_output, att_seq = None, None, None
        dec_state = [] if get_dec_state else None

        # Encode
        encode_feature, encode_len = self.encoder(audio_feature, feature_len)

        # CTC based decoding
        if self.enable_ctc:
            ctc_output = F.log_softmax(self.ctc_layer(encode_feature), dim=-1)

        # Attention based decoding
        if self.enable_att:
            # Init (init char = <SOS>, reset all rnn state and cell)
            self.decoder.init_state(bs)
            self.attention.reset_mem()
            last_char = self.pre_embed(torch.zeros(
                (bs), dtype=torch.long, device=encode_feature.device))
            att_seq, output_seq = [], []

            # Preprocess data for teacher forcing
            if teacher is not None:
                teacher = self.embed_drop(self.pre_embed(teacher))

            # Decode
            for t in range(decode_step):
                # Attend (inputs current state of first layer, encoded features)
                attn, context = self.attention(
                    self.decoder.get_query(), encode_feature, encode_len)
                # Decode (inputs context + embedded last character)
                decoder_input = torch.cat([last_char, context], dim=-1)
                cur_char, d_state = self.decoder(decoder_input)
                # Prepare output as input of next step
                if (teacher is not None):
                    # Training stage
                    if (tf_rate == 1) or (torch.rand(1).item() <= tf_rate):
                        # teacher forcing
                        last_char = teacher[:, t, :]
                    else:
                        # self-sampling (replace by argmax may be another choice)
                        with torch.no_grad():
                            if (emb_decoder is not None) and emb_decoder.apply_fuse:
                                _, cur_prob = emb_decoder(
                                    d_state, cur_char, return_loss=False)
                            else:
                                cur_prob = cur_char.softmax(dim=-1)
                            sampled_char = Categorical(cur_prob).sample()
                        last_char = self.embed_drop(
                            self.pre_embed(sampled_char))
                else:
                    # Inference stage
                    if (emb_decoder is not None) and emb_decoder.apply_fuse:
                        _, cur_char = emb_decoder(
                            d_state, cur_char, return_loss=False)
                    # argmax for inference
                    last_char = self.pre_embed(torch.argmax(cur_char, dim=-1))

                # save output of each step
                output_seq.append(cur_char)
                att_seq.append(attn)
                if get_dec_state:
                    dec_state.append(d_state)

            att_output = torch.stack(output_seq, dim=1)  # BxTxV
            att_seq = torch.stack(att_seq, dim=2)       # BxNxDtxT
            if get_dec_state:
                dec_state = torch.stack(dec_state, dim=1)

        return ctc_output, encode_len, att_output, att_seq, dec_state


class Decoder(nn.Module):
    ''' Decoder (a.k.a. Speller in LAS) '''
    # ToDo:　More elegant way to implement decoder

    def __init__(self, input_dim, vocab_size, module, dim, layer, dropout):
        super(Decoder, self).__init__()
        self.in_dim = input_dim
        self.layer = layer
        self.dim = dim
        self.dropout = dropout

        # Init
        assert module in ['LSTM', 'GRU'], NotImplementedError
        self.hidden_state = None
        self.enable_cell = module == 'LSTM'

        # Modules
        self.layers = getattr(nn, module)(
            input_dim, dim, num_layers=layer, dropout=dropout, batch_first=True)
        self.char_trans = nn.Linear(dim, vocab_size)
        self.final_dropout = nn.Dropout(dropout)

    def init_state(self, bs):
        ''' Set all hidden states to zeros '''
        device = next(self.parameters()).device
        if self.enable_cell:
            self.hidden_state = (torch.zeros((self.layer, bs, self.dim), device=device),
                                 torch.zeros((self.layer, bs, self.dim), device=device))
        else:
            self.hidden_state = torch.zeros(
                (self.layer, bs, self.dim), device=device)
        return self.get_state()

    def set_state(self, hidden_state):
        ''' Set all hidden states/cells, for decoding purpose'''
        device = next(self.parameters()).device
        if self.enable_cell:
            self.hidden_state = (hidden_state[0].to(
                device), hidden_state[1].to(device))
        else:
            self.hidden_state = hidden_state.to(device)

    def get_state(self):
        ''' Return all hidden states/cells, for decoding purpose'''
        if self.enable_cell:
            return (self.hidden_state[0].cpu(), self.hidden_state[1].cpu())
        else:
            return self.hidden_state.cpu()

    def get_query(self):
        ''' Return state of all layers as query for attention '''
        if self.enable_cell:
            return self.hidden_state[0].transpose(0, 1).reshape(-1, self.dim*self.layer)
        else:
            return self.hidden_state.transpose(0, 1).reshape(-1, self.dim*self.layer)

    def forward(self, x):
        ''' Decode and transform into vocab '''
        if not self.training:
            self.layers.flatten_parameters()
        x, self.hidden_state = self.layers(x.unsqueeze(1), self.hidden_state)
        x = x.squeeze(1)
        char = self.char_trans(self.final_dropout(x))
        return char, x


class Attention(nn.Module):
    ''' Attention mechanism
        please refer to http://www.aclweb.org/anthology/D15-1166 section 3.1 for more details about Attention implementation
        Input : Decoder state                      with shape [batch size, decoder hidden dimension]
                Compressed feature from Encoder    with shape [batch size, T, encoder feature dimension]
        Output: Attention score                    with shape [batch size, num head, T (attention score of each time step)]
                Context vector                     with shape [batch size, encoder feature dimension]
                (i.e. weighted (by attention score) sum of all timesteps T's feature) '''

    def __init__(self, v_dim, q_dim, mode, dim, num_head, temperature, v_proj,
                 loc_kernel_size, loc_kernel_num):
        super(Attention, self).__init__()

        # Setup
        self.v_dim = v_dim
        self.dim = dim
        self.mode = mode.lower()
        self.num_head = num_head

        # Linear proj. before attention
        self.proj_q = nn.Linear(q_dim, dim*num_head)
        self.proj_k = nn.Linear(v_dim, dim*num_head)
        self.v_proj = v_proj
        if v_proj:
            self.proj_v = nn.Linear(v_dim, v_dim*num_head)

        # Attention
        if self.mode == 'dot':
            self.att_layer = ScaleDotAttention(temperature, self.num_head)
        elif self.mode == 'loc':
            self.att_layer = LocationAwareAttention(
                loc_kernel_size, loc_kernel_num, dim, num_head, temperature)
        else:
            raise NotImplementedError

        # Layer for merging MHA
        if self.num_head > 1:
            self.merge_head = nn.Linear(v_dim*num_head, v_dim)

        # Stored feature
        self.key = None
        self.value = None
        self.mask = None

    def reset_mem(self):
        self.key = None
        self.value = None
        self.mask = None
        self.att_layer.reset_mem()

    def set_mem(self, prev_attn):
        self.att_layer.set_mem(prev_attn)

    def forward(self, dec_state, enc_feat, enc_len):

        # Preprecessing
        bs, ts, _ = enc_feat.shape
        query = torch.tanh(self.proj_q(dec_state))
        query = query.view(bs, self.num_head, self.dim).view(
            bs*self.num_head, self.dim)  # BNxD

        if self.key is None:
            # Maskout attention score for padded states
            self.att_layer.compute_mask(enc_feat, enc_len.to(enc_feat.device))

            # Store enc state to lower computational cost
            self.key = torch.tanh(self.proj_k(enc_feat))
            self.value = torch.tanh(self.proj_v(
                enc_feat)) if self.v_proj else enc_feat  # BxTxN

            if self.num_head > 1:
                self.key = self.key.view(bs, ts, self.num_head, self.dim).permute(
                    0, 2, 1, 3)  # BxNxTxD
                self.key = self.key.contiguous().view(bs*self.num_head, ts, self.dim)  # BNxTxD
                if self.v_proj:
                    self.value = self.value.view(
                        bs, ts, self.num_head, self.v_dim).permute(0, 2, 1, 3)  # BxNxTxD
                    self.value = self.value.contiguous().view(
                        bs*self.num_head, ts, self.v_dim)  # BNxTxD
                else:
                    self.value = self.value.repeat(self.num_head, 1, 1)

        # Calculate attention
        context, attn = self.att_layer(query, self.key, self.value)
        if self.num_head > 1:
            context = context.view(
                bs, self.num_head*self.v_dim)    # BNxD  -> BxND
            context = self.merge_head(context)  # BxD

        return attn, context


class Encoder(nn.Module):
    ''' Encoder (a.k.a. Listener in LAS)
        Encodes acoustic feature to latent representation, see config file for more details.'''

    def __init__(self, input_size, vgg, module, bidirection, dim, dropout, layer_norm, proj, sample_rate, sample_style):
        super(Encoder, self).__init__()

        # Hyper-parameters checking
        self.vgg = vgg
        self.sample_rate = 1
        assert len(sample_rate) == len(dropout), 'Number of layer mismatch'
        assert len(dropout) == len(dim), 'Number of layer mismatch'
        num_layers = len(dim)
        assert num_layers >= 1, 'Encoder should have at least 1 layer'

        # Construct model
        module_list = []
        input_dim = input_size

        if vgg:
            vgg_extractor = VGGExtractor(input_size)
            module_list.append(vgg_extractor)
            input_dim = vgg_extractor.out_dim
            self.sample_rate = self.sample_rate*4

        if module in ['LSTM', 'GRU']:
            for l in range(num_layers):
                module_list.append(RNNLayer(input_dim, module, dim[l], bidirection, dropout[l], layer_norm[l],
                                            sample_rate[l], sample_style, proj[l]))
                input_dim = module_list[-1].out_dim
                self.sample_rate = self.sample_rate*sample_rate[l]
        else:
            raise NotImplementedError

        self.in_dim = input_size
        self.out_dim = input_dim
        self.layers = nn.ModuleList(module_list)

    def forward(self, input_x, enc_len):
        for _, layer in enumerate(self.layers):
            input_x, enc_len = layer(input_x, enc_len)
        return input_x, enc_len
