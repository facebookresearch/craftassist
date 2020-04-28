"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import math

import torch
import torch.nn as nn


class HighwayNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(HighwayNetwork, self).__init__()
        self.gate_proj = nn.Linear(in_dim, out_dim)
        self.lin_proj = nn.Linear(in_dim, out_dim)
        self.nonlin_proj = nn.Linear(in_dim, out_dim)
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)
            else:
                torch.nn.init.constant_(p, 0)

    def forward(self, x):
        gate = torch.sigmoid(self.gate_proj(x) - 2)
        lin = self.lin_proj(x)
        nonlin = torch.relu(self.nonlin_proj(x))
        res = gate * nonlin + (1 - gate) * lin
        return res


# simple sentence embedding usinf pre-trained fastText
# for debugging
class FastTextSentenceEmbedding(nn.Module):
    def __init__(self, dim, l_dim, n_words, ft_embeds, init_embeds=True):
        super(FastTextSentenceEmbedding, self).__init__()
        self.dim = dim
        self.ft_dim = 0 if ft_embeds is None else 300
        self.l_dim = l_dim
        self.ft_embed = False if ft_embeds is None else nn.Embedding.from_pretrained(ft_embeds)
        if l_dim > 0:
            self.learn_embed = nn.Embedding(n_words, l_dim)
            for p in self.learn_embed.parameters():
                torch.nn.init.constant_(p, 0)
        self.proj = HighwayNetwork(self.ft_dim + self.l_dim, self.dim)
        for p in self.proj.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)
            else:
                torch.nn.init.constant_(p, 0)
        if l_dim > 0:
            if self.ft_embed:
                for p in self.learn_embed.parameters():
                    torch.nn.init.constant_(p, 0)
            else:
                for p in self.learn_embed.parameters():
                    if p.dim() > 1:
                        torch.nn.init.xavier_normal_(p)
                    else:
                        torch.nn.init.constant_(p, 0)

    def forward(self, sent_ids, sent_mask=None, sent_len=None):
        if self.ft_embed:
            ft_embed = self.ft_embed(sent_ids).detach()
        if self.l_dim > 0:
            l_embed = self.learn_embed(sent_ids)
        if self.ft_embed and self.l_dim > 0:
            pre_embed = torch.cat([ft_embed, l_embed], dim=-1)
        elif self.ft_embed:
            pre_embed = ft_embed
        else:
            pre_embed = l_embed
        return self.proj(pre_embed)


# Sinusoidal position encodings
class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout, max_len=320):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].detach()
        return self.dropout(x)


# convolutional sentence encoder
class SentenceConvEncoder(nn.Module):
    def __init__(self, word_embed, dim, k, n_layers, dropout=0.1):
        super(SentenceConvEncoder, self).__init__()
        self.word_embed = word_embed
        self.conv_layers = nn.ModuleList(
            [nn.Conv1d(dim, 2 * dim, 2 * k + 1, padding=k) for _ in range(n_layers)]
        )
        self.glu = nn.GLU()
        self.do = nn.Dropout(dropout)
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)
            else:
                torch.nn.init.constant_(p, 0)
        # don't initialize PE!
        self.pe = PositionalEncoding(dim, dropout)

    def forward(self, sent_ids, sent_mask, sent_len):
        x = self.word_embed(sent_ids)
        x = self.do(x)
        for layer in self.conv_layers:
            residual = x
            x = layer(x.transpose(1, 2)).transpose(1, 2)
            x = self.glu(x) + residual
            x = x.masked_fill(sent_mask.unsqueeze(-1) == 0, 0)
            x = self.do(x)
        return x


# BiGRU sentence encoder
class SentenceBiGRUEncoder(nn.Module):
    def __init__(self, word_embed, dim, n_layers, dropout=0.1):
        super(SentenceBiGRUEncoder, self).__init__()
        self.word_embed = word_embed
        self.bigru = nn.GRU(
            dim, dim, n_layers, batch_first=True, bidirectional=True, dropout=dropout
        )
        self.final_proj = nn.Linear(2 * dim, dim)
        self.do = nn.Dropout(dropout)
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)
            else:
                torch.nn.init.constant_(p, 0)

    def forward(self, sent_ids, sent_mask, sent_len):
        x = self.word_embed(sent_ids)
        x = self.do(x)
        lengths = sent_len
        # TODO this is a temporary hotfix for unsorted padded sequence
        if sent_mask.shape[0] > 1:
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
        else:
            packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        packed_h, _ = self.bigru(packed_x)
        h, _ = nn.utils.rnn.pad_packed_sequence(packed_h, batch_first=True)
        h = self.do(h)
        h = self.final_proj(h)
        return h
