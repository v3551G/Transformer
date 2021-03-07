# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import copy
import seaborn
seaborn.set_context(context="talk")
#matplotlib inline

from attention_layer import MultiHeadAttention
from feedforward_layer import PositionwiseFeedForward
from position_encoding_layer import PositionalEncoding
from embedding_layer import Embeddings

from encoder import Encoder, EncoderLayer
from decoder import Decoder, DecoderLayer



'''
   masterqkk, masterqkk@Outlook.com， 20210306
'''

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    # build base layer
    attention_layer = MultiHeadAttention(h, d_model)
    feed_forward_layer = PositionwiseFeedForward(d_model, d_ff, dropout)
    position_encoding_layer = PositionalEncoding(d_model, dropout)

    # build building block of Encoder/Decoder
    encoder_block = EncoderLayer(d_model, c(attention_layer), c(feed_forward_layer), dropout)
    decoder_block = DecoderLayer(d_model, c(attention_layer), c(attention_layer), c(feed_forward_layer), dropout)
    #

    # build module
    encoder = Encoder(encoder_block, N)
    decoder = Decoder(decoder_block, N)
    src_embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position_encoding_layer))
    tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), c(position_encoding_layer))
    generator = Generator(d_model, tgt_vocab)

    model = EncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator)

    # Initialize model parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return model


class EncoderDecoder(nn.Module):
    '''
    Encoder-Decoder architecture.
    '''
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        encoder_out = self.encoder(src, src_mask)
        return self.decoder(encoder_out, src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        # Note: mask is done on the embedded vector
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt, tgt_mask)


class Generator(nn.Module):
    '''
    The last operation in Decoder.
    '''
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.projector = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.projector(x), dim=-1)


