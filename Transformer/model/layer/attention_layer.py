# -*- coding: utf-8 -*-
import math
import torch
import torch.nn.functional as F
import torch.nn as nn

from tools import clones

'''
    Attention layer
    masterqkk, masterqkk@Outlook.com， 20210306
'''


def attention(query, keys, values, mask=None, dropout=None):
    '''
    Scaled dot product attention
    :param query:
    :param keys:
    :param values:
    :param mask:
    :param dropout:
    :return:
    '''
    d_k = query.size(-1)
    scores = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attention = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attention = dropout(p_attention)

    return torch.matmul(p_attention, values), p_attention


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        '''
        :param h: the number of heads
        :param d_model: the initial dimension.
        :param dropout:
        '''
        super(MultiHeadAttention, self).__init__()
        assert(d_model % h == 0)
        self.d_k = d_model // h
        self.h = h
        # including the three pre projection layer for query, key and values plus the final linear projection
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attention = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, keys, values, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        num_batchs = query.size(0)
        query, keys, values = [l(x).view(num_batchs, -1, self.h, self.d_k).transpose(1,2) for l, x in zip(self.linears, (query, keys, values))]

        x, self.attention = attention(query, keys, values, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(num_batchs, -1, self.h * self.d_k)

        return self.linears[-1](x)


