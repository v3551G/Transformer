# -*- coding: utf-8 -*-
import torch.nn as nn

from tools import clones, LayerNorm, SublayerConnection

'''
   masterqkk, masterqkk@Outlook.com， 20210306
'''


class Decoder(nn.Module):
    '''
    Definitation of Transformer_Decoder
    '''
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.normalizer = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.normalizer(x)


class DecoderLayer(nn.Module):
    '''
    Building block of Decoder
    '''
    def __init__(self, size, self_attention, src_attention, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention = self_attention
        self.src_attention = src_attention
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(size, dropout), 3)
        self.size = size

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayers[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.sublayers[1](x, lambda x: self.src_attention(x, m, m, src_mask))
        return self.sublayers[2](x, self.feed_forward)


