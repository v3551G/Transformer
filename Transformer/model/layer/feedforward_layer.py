# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F

'''
    Positionwise feed forward layer
    masterqkk, masterqkk@Outlook.com， 20210306
'''


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, drop_out=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x):
        return self.linear_2(self.dropout(F.relu(self.linear_1(x))))

