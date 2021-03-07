# -*- coding: utf-8 -*-
import copy
import math
import numpy as np

import torch
import torch.nn as nn

'''
   masterqkk, masterqkk@Outlook.com， 20210306
'''


class LayerNorm(nn.Module):
    '''
    Layer normalization module
    '''
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keep_dim=True)
        std = x.std(-1, keep_dim=True)
        return self.a_2 * (x-mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    '''
    Wrapper of sub_layer(attention layer or feed forward layer)
    residual connection followed by a layer normalization, i.e. x ->  layer_norm(x + sublayer(x)),
    for simlication, it's implemented as, x ->  x + sublayer(layer_norm(x)))
    '''
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.normalizer = LayerNorm(size)
        self.dropout_op = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout_op(sublayer(self.normalizer(x)))


def clones(module, N):
    '''produce N independent, identical layers through deepcopy'''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    attention_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attention_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0