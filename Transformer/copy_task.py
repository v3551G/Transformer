# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable

from data_loader import data_gen
from process import LabelSmoothing, NoamOpt, run_epoch
from loss import SimpleLossCompute
from model.model_builder import make_model
from model.tools import subsequent_mask

'''
    A simple copy-task. Given a random set of input symbols from a small vocabulary, 
    the goal is to generate back those same symbols.
    masterqkk@outlook.com, 20210307
'''


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)

    for i in range(max_len-1):
        out = model.decode(memory, src_mask, Variable(ys), Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


if __name__ == '__main__':
    V = 11
    batch_size = 30
    num_batches = 20
    num_epoch = 10

    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(src_vocab=V, tgt_vocab=V, N=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    model_opt = NoamOpt(model_size=model.src_embed[0].d_model, factor=1, warmup=400, optimizer=optimizer)

    for epoch in range(num_epoch):
        model.train()
        loss_compute = SimpleLossCompute(generator=model.generator, criterion=criterion, opt=model_opt)
        run_epoch(data_iter=data_gen(v=V, batch=batch_size, num_batches=num_batches), model=model, loss_compute=loss_compute)

        model.eval()
        loss_compute2 = SimpleLossCompute(generator=model.generator, criterion=criterion, opt=None)
        print(run_epoch(data_iter=data_gen(v=V, batch=batch_size, num_batches=5), mode=model, loss_compute=loss_compute2))
    #
    model.eval()
    src = Variable(torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))
    src_mask = Variable(torch.ones(1, 1, 10))

    ys = greedy_decode(model, src, src_mask, max_len=10, start_symbol=1)
    print(ys)