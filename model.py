import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import math
import time
import warnings
import os
import datetime
import scipy

import random
warnings.simplefilter('ignore')
torch.manual_seed(0)
np.random.seed(0)

class Transformer_ours(nn.Module):
    def __init__(self, 
        in_features: int, # num of features to represent stock information
        out_features: int, # num of features to represent the predicted stock information, default value will be: 1
        enc_seq_len: int,
        dec_seq_len: int,
        d_model: int,  
        n_encoder_layers: int,
        n_decoder_layers: int,
        n_heads: int,
        dropout_encoder: float=0.2, 
        dropout_decoder: float=0.2,
        dropout_pos_enc: float=0.2,
        dropout_pos_dec: float=0.2,
        dim_feedforward_encoder: int=2048,
        dim_feedforward_decoder: int=2048,
        conv_size: int = 10,
        long_dependency_multiplier: int = 4
        ): 

        super().__init__() 

        self.n_heads = n_heads
        self.enc_seq_len = enc_seq_len
        self.dec_seq_len = dec_seq_len
        self.long_dependency_multiplier = long_dependency_multiplier
        self.in_features = in_features
        # after adding long dependency, #cols becomes 1.5*in_features
        self.encoder_input_layer = nn.Linear(in_features=int(in_features*1.5), out_features=d_model)
        self.decoder_input_layer = nn.Linear(in_features=in_features, out_features=d_model)  
        
        self.linear_mapping = nn.Linear(in_features=d_model,out_features=out_features)
        self.long_dependency_compression = nn.Linear(in_features = int(in_features*1.5), out_features = int(in_features*0.5))
        
        # self.positional_encoding_layer = Time2Vector(seq_len=enc_seq_len, out_features=d_model, dropout=dropout_pos_enc)
        # self.positional_decoding_layer = Time2Vector(seq_len=dec_seq_len, out_features=d_model, dropout=dropout_pos_dec)
        
        encoder_layer = MyEncoder(
            d_model=d_model, 
            nheads=n_heads,
            dim_feedforward_encoder=dim_feedforward_encoder,
            dropout_encoder=dropout_encoder,
            conv_size=conv_size,
            )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,num_layers=n_encoder_layers, norm=None)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_decoder,
            dropout=dropout_decoder,
            batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer,num_layers=n_decoder_layers, norm=None)
    
    def forward(self, src, tgt):    
        encoder_seq_len_true = int(self.enc_seq_len/self.long_dependency_multiplier)
        '''add long dependency'''
        new_src = torch.empty(len(src), encoder_seq_len_true, int(1.5*self.in_features))
        history = torch.zeros(encoder_seq_len_true, int(self.in_features*0.5))
        for i in range(len(src)):
            for j in range(self.long_dependency_multiplier):
                src_temp = src[i][j*encoder_seq_len_true: (j+1)*encoder_seq_len_true]
                src_temp = torch.concat([src_temp, history], dim = -1)
                history = self.long_dependency_compression(src_temp)
            new_src[i] = src_temp      
        new_src = self.encoder_input_layer(new_src)
        new_src = self.encoder(src=new_src)      
        tgt = self.decoder_input_layer(tgt)
        tgt = self.decoder(tgt=tgt, memory=new_src)

        decoder_output= self.linear_mapping(tgt)

        return decoder_output.squeeze()


class MyEncoder(nn.Module):
    def __init__(self, d_model: int, nheads: int, dim_feedforward_encoder: int = 2048, dropout_encoder: float = 0.2, conv_size: int = 10):
        super(MyEncoder, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nheads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            batch_first=True)

        self.convolution_layer = nn.Conv2d(1,1,(conv_size,1),stride = 1,bias = False)
        self.combine_encoder_conv = nn.Linear(in_features = 2*d_model, out_features = d_model)
        self.conv_size = conv_size

    def forward(self, x, src_mask: Tensor= None, src_key_padding_mask: Tensor = None): 
        encoder_layer_temp = self.encoder_layer(src = x)   
        x_pad = F.pad(x, (0,0, self.conv_size - 1,0)) # add padding so that the output of a convolution layer and encoder layer are the same
        x_pad = torch.unsqueeze(x_pad, dim = 1)
        conv_layer_temp = self.convolution_layer(x_pad)
        conv_layer_temp = torch.squeeze(conv_layer_temp, dim = 1) 
        temp = torch.concat([encoder_layer_temp, conv_layer_temp], dim = -1)
        x = self.combine_encoder_conv(temp)
        return x
