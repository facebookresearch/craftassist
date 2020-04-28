"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import copy
import math

import torch
import torch.nn as nn

##### Utility modules
# Produce n identical layers.
def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def xavier_init(module, mul=1.0):
    for p in module.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_normal_(p, mul)
        else:
            torch.nn.init.constant_(p, 0)


# Compute 'Scaled Dot Product Attention'
def attention(query, key, value, mask=None, dropout=None):
    dim = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dim)
    if mask is not None:
        scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
    p_attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, self.h * self.d_k)
        return self.linears[-1](x)

    def initialize(self, mul):
        for lin in self.linears:
            xavier_init(lin, mul)


# node sub-modules

# NodeSentRep attends to sentence representation (and possibly previously visited nodes)
# to get contextual representation vector for current node
# used to predict the presence of an internal node or the value of a categorical node
class NodeSentenceRep(nn.Module):
    def __init__(self, tree, sent_attn, tree_attn, rec_cell):
        super(NodeSentenceRep, self).__init__()
        self.dim = tree.dim
        self.do = nn.Dropout(tree.dropout)
        # attending over the tree context first
        self.tree_attn = tree_attn
        self.sent_attn = sent_attn
        self.rec_proj = nn.Linear(2 * self.dim, self.dim)
        self.rec_cell = rec_cell

    def initialize(self, mul):
        self.tree_attn.initialize(mul)
        self.sent_attn.initialize(mul)
        xavier_init(self.rec_cell, mul)
        xavier_init(self.rec_proj, mul)

    def forward(self, node_vec, sentence_rep, tree_context):
        sent_vecs, sent_mask = sentence_rep
        prev_h, parent_h, context_rep, context_mask = tree_context
        if prev_h is not None:
            node_vec = node_vec + prev_h
        if context_rep is not None:
            node_rep = self._apply_tree_attn(node_vec, context_rep, context_mask)
        else:
            node_rep = node_vec
        node_rep = self._apply_sen_attn(node_rep, sent_vecs, sent_mask)
        return node_rep

    # our (sentence + tree)-attention baseline
    def _apply_tree_attn(self, node_vec, context_rep, context_mask):
        # if recursive:  attend over provided context
        #                to compute node representaiton
        # else:  just use node parameter
        # attend over context
        context_h = self.tree_attn(node_vec, context_rep, context_rep, mask=context_mask)
        node_tree_rep = self.do(context_h + node_vec)
        return node_tree_rep

    # inplementing Seq2Tree from Dong & Lapata
    def _apply_sen_attn(self, node_vec, sent_rep, sent_mask):
        # predict presence and value based on previous hidden space
        node_rep = self.sent_attn(node_vec, sent_rep, sent_rep, mask=sent_mask)
        node_pred = self.do(node_vec + node_rep)
        return node_pred

    def _apply_recurrence(self, node_rep, prev_h, parent_h, on_mask):
        # update hidden space based on node vector and previous and parent hidden space
        if parent_h is None:
            node_vec = node_rep
        else:
            node_vec = self.rec_proj(torch.cat([node_rep, parent_h], dim=-1))
        node_h = self.rec_cell(node_vec, prev_h)
        # only update hidden state for active nodes
        node_h = on_mask[:, None] * node_h + (1 - on_mask[:, None]) * prev_h
        return node_h


# Various internal node and leaf prediction heads
# sigmoid for presence scores
class PredictPresence(nn.Module):
    def __init__(self, dim):
        super(PredictPresence, self).__init__()
        self.score_pres = nn.Linear(dim, 1)

    def initialize(self, mul):
        xavier_init(self.score_pres, mul)

    def forward(self, x):
        scores = self.score_pres(x).squeeze(dim=-1)
        return scores


# softmax over class scores
class PredictCategorySingle(nn.Module):
    def __init__(self, dim, n_vals):
        super(PredictCategorySingle, self).__init__()
        self.score_vals = nn.Linear(dim, n_vals)

    def initialize(self, mul):
        xavier_init(self.score_vals, mul)

    def forward(self, x):
        scores = self.score_vals(x)
        scores = torch.log_softmax(scores, dim=-1)
        return scores


# sigmoid for each class score
class PredictCategorySet(nn.Module):
    def __init__(self, dim, n_vals):
        super(PredictCategorySet, self).__init__()
        self.score_vals = nn.Linear(dim, n_vals)

    def initialize(self, mul):
        xavier_init(self.score_vals, mul)

    def forward(self, x):
        scores = self.score_vals(x)
        return scores


# softmax over indexes for start and end
class PredictSpanSingle(nn.Module):
    def __init__(self, dim):
        super(PredictSpanSingle, self).__init__()
        self.score_start = nn.Linear(dim, dim)
        self.score_end = nn.Linear(dim, dim)

    def initialize(self, mul):
        xavier_init(self.score_start, mul)
        xavier_init(self.score_end, mul)

    def forward(self, x, sentence_rep):
        sen_vecs, sen_mask = sentence_rep
        sen_start = self.score_start(sen_vecs)  # B x T x D
        scores_start = torch.matmul(sen_start, x.unsqueeze(-1)).squeeze(-1)  # B x T
        scores_start = scores_start.masked_fill(sen_mask == 0, -1e6)
        sen_end = self.score_end(sen_vecs)  # B x T x D
        scores_end = torch.matmul(sen_end, x.unsqueeze(-1)).squeeze(-1)  # B x T
        scores_end = scores_end.masked_fill(sen_mask == 0, -1e6)
        scores = scores_start.unsqueeze(-1) + scores_end.unsqueeze(-2)  # B x T x T
        bs, t = scores.shape[:2]
        scores += torch.tril(-1e6 * torch.ones(t, t), diagonal=-1).unsqueeze(0).type_as(sen_vecs)
        scores = scores.view(bs, -1)
        scores = torch.log_softmax(scores, dim=-1)
        scores = scores.view(bs, t, t)
        return scores


# BIO tagging
# (no transition scores in thids implementation, TODO)
def log_sum_exp(bx):
    bm = bx.max(dim=-1)[0]
    return bm + torch.log(torch.exp(bx - bm.unsqueeze(-1)).sum(dim=-1))


class PredictSpanSet(nn.Module):
    def __init__(self, dim):
        super(PredictSpanSet, self).__init__()
        self.score_b = nn.Linear(dim, dim)
        self.score_i = nn.Linear(dim, dim)
        self.score_o = nn.Linear(dim, dim)

    def initialize(self, mul):
        xavier_init(self.score_b, mul)
        xavier_init(self.score_i, mul)
        xavier_init(self.score_o, mul)

    def forward(self, x, sentence_rep):
        sen_vecs, sen_mask = sentence_rep
        sen_b = self.score_b(sen_vecs)  # B x T x D
        scores_b = torch.matmul(sen_b, x.unsqueeze(-1)).squeeze(-1)  # B x T
        scores_b = scores_b.masked_fill(sen_mask == 0, -1e6)
        sen_i = self.score_i(sen_vecs)  # B x T x D
        scores_i = torch.matmul(sen_i, x.unsqueeze(-1)).squeeze(-1)  # B x T
        scores_i = scores_i.masked_fill(sen_mask == 0, -1e6)
        sen_o = self.score_o(sen_vecs)  # B x T x D
        scores_o = torch.matmul(sen_o, x.unsqueeze(-1)).squeeze(-1)  # B x T
        scores_o = scores_o.masked_fill(sen_mask == 0, 0)
        scores_bio = torch.cat(
            [scores_b.unsqueeze(-1), scores_i.unsqueeze(-1), scores_o.unsqueeze(-1)], dim=-1
        )  # B x T x 3
        # alpha beta recursion
        bs, bt = scores_o.shape
        forward_scores = torch.zeros(bs, bt + 1).type_as(scores_o)
        backward_scores = torch.zeros(bs, bt + 1).type_as(scores_o)
        for t in range(1, bt + 1):
            forward_scores[:, t] = log_sum_exp(
                forward_scores[:, t - 1].unsqueeze(dim=-1) + scores_bio[:, t - 1]
            )
            backward_scores[:, -t - 1] = log_sum_exp(
                backward_scores[:, -t].unsqueeze(dim=-1) + scores_bio[:, -t]
            )
        full_scores = forward_scores[:, -1]
        span_scores = torch.zeros(bs, bt, bt).type_as(sen_vecs)
        for s in range(bt):
            for e in range(s, bt):
                span_scores[:, e, s] = -1e3
                span_scores[:, s, e] = (
                    scores_b[:, s]
                    + scores_i[:, s:e].sum(dim=-1)
                    + forward_scores[:, s]
                    + backward_scores[:, e + 1]
                    - full_scores
                )
                if e < bt:
                    span_scores[:, s, e] += scores_o[:, e]
        return span_scores
