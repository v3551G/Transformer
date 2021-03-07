# -*- coding: utf-8 -*-
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
from loss import LabelSmoothing

'''
    masterqkk, masterqkk@outlook.com
'''


class NoamOpt:
    '''
    Wrapper of optimizer that implements rate
    '''

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        '''
        rate = factor * d_model^{-0.5} * min(step_num^{-0.5}, step_nnum * warmup^{-1.5})
        '''
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    return NoamOpt(model.src_embed[0].d_model, 2, 4000, optimizer)


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)

    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


def run_epoch(data_iter, model, loss_compute):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0

    for i, batch in enumerate(data_iter):
        output = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(output, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens

        tokens += batch.ntokens

        if i % 50 == 1:
            elapsed = time.time() - start
            print('Epoch step: %d, Loss: %f, Tokens per Sec; %f' % (i, loss / batch.ntokens, tokens / elapsed))

            start = time.time()
            tokens = 0
    return total_loss / total_tokens


if __name__ == '__main__':
    # Three settings of the learning_rate
    opts = [NoamOpt(512, 1, 4000, None),
            NoamOpt(512, 1, 8000, None),
            NoamOpt(256, 1, 4000, None)]
    plt.figure()
    plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
    plt.legend(['512:4000', '512:8000', '256:4000'])
    plt.show()

    # Label smoothing
    crit = LabelSmoothing(5, 0, 0.4)
    predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                                 [0, 0.2, 0.7, 0.1, 0],
                                 [0, 0.2, 0.7, 0.1, 0]])
    v = crit(x=Variable(predict.log()), target=Variable(torch.LongTensor([2, 1, 0])))
    plt.imshow(crit.true_dist)
    plt.show()

    ## label smoothing starts to penalize the model if it's very confidence to a choice
    crit = LabelSmoothing(5, 0, 0.1)

    def loss(x):
        d = x + 3 * 1
        predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d]])
        return crit(x=Variable(predict.log()), target=Variable(torch.LongTensor([1]))).data

    x = np.arange(1, 100)
    y = [loss(x) for x in range(1, 100)]
    plt.figure()
    plt.plot(x, y)
    plt.xlabel('uncertainity -> certainity')
    plt.ylabel('loss/penality')
    plt.show()
