# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable

'''
   masterqkk@outlook.com
'''


class SimpleLossCompute:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()

        return loss.data[0] * norm


class LabelSmoothing(nn.Module):
    '''
    Label smoothing based on KL divergence
    '''

    def __init__(self, size, padding_idx, smoothing=0.0):
        '''
        padding_idx:index of padding position
        smoothing:
        '''
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert (x.size(1) == self.size)
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2)) # 2: the ground-truth and padding
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        #if mask.dim() > 0:
        if mask.numel() > 0:
            xx = mask.data
            true_dist.index_fill_(dim=0, index=mask.squeeze(), value=0.0) # fill position-index in dim-th with value
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


