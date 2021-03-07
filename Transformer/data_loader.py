# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
from model.tools import subsequent_mask

'''
    masterqkk@outlook.com, 20210307
'''


def data_gen(v, batch, num_batches):
    for i in range(num_batches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 1] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)


class Batch:
    '''
    Object for holding a batch of data with mask during training
    '''

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)

        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask
