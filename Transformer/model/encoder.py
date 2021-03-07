# -*- coding: utf-8 -*-

import torch.nn as nn

from tools import clones, LayerNorm, SublayerConnection

'''
   masterqkk, masterqkk@Outlook.com， 20210306
'''


class Encoder(nn.Module):
    '''
    Defination of Transformer_Encoder
    '''
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        # the last layer normalization of the Encoder
        self.normalizer = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.normalizer(x)


class EncoderLayer(nn.Module):
    '''
    Building block of Encoder, which is composed of two sub_layers: attention layer and feed forward layer, these two sub_layers are followed by residual connection with layer normalization, which is defined as SublayerConnection
    '''
    def __init__(self, size, self_attention, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attention =self_attention
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayers[0](x, lambda x: self.self_attention(x, x, x, mask))
        return self.sublayers[1](x, self.feed_forward)


