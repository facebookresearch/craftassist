"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import json
import math
import pickle

import torch
import torch.nn as nn

from .action_tree import *
from .model_utils import *
from .my_optim import *
from .sentence_encoder import *

##### Define tree modules

# Each node module computes a contextual node representation
# and predicts whether the node is present, as we as its
# categorical or span value for leaves
class NodePrediction(nn.Module):
    def __init__(self, node_sen_rep, node, args):
        super(NodePrediction, self).__init__()
        self.dim = args.model_dim
        self.dropout = nn.Dropout(args.dropout)
        self.node_type = node.node_type
        self.node_vec = nn.Parameter(torch.zeros(self.dim))
        self.node_sen_rep = node_sen_rep
        self.node_score_pres = PredictPresence(self.dim)
        # have to define module outside of loop for pytorch reasons
        n_values = len(node.choices) if "categorical" in self.node_type else 1
        self.nsv_s_sin = PredictSpanSingle(self.dim)
        self.nsv_s_set = PredictSpanSet(self.dim)
        self.nsv_c_sin = PredictCategorySingle(self.dim, n_values)
        self.nsv_c_set = PredictCategorySet(self.dim, n_values)
        self.nvv = nn.Embedding(n_values, self.dim)
        self.module_list = nn.ModuleList(
            [
                self.node_score_pres,
                self.nsv_s_sin,
                self.nsv_s_set,
                self.nsv_c_sin,
                self.nsv_c_set,
                self.nvv,
            ]
        )
        if self.node_type == "internal":
            self.node_score_val = None
        elif self.node_type == "span-single":
            self.node_score_val = self.nsv_s_sin
        elif self.node_type == "span-set":
            self.node_score_val = self.nsv_s_set
        elif self.node_type == "categorical-single":
            self.node_val_vec = self.nvv
            self.node_score_val = self.nsv_c_sin
        elif self.node_type == "categorical-set":
            self.node_val_vec = self.nvv
            self.node_score_val = self.nsv_c_set
        else:
            raise NotImplementedError

    def forward(self, sentence_rep, tree_context, recursion, active, labels):
        # compute node representation to be used in prediction
        sent_rep, sent_mask = sentence_rep
        node_vec = self.node_vec[None, :].expand(active.shape[0], self.dim).type_as(sent_rep)
        node_rep = self.node_sen_rep(node_vec, sentence_rep, tree_context)
        node_rep = self.dropout(node_rep)
        if recursion == "seq2tree-rec":
            node_x = node_vec
        else:
            node_x = node_rep
        # predict whether the node is on
        # uses labels if available or predicted value otherwise
        # for recurrence / recursion update
        pres_score = self.node_score_pres(node_rep)
        if labels is None:
            on_mask = ((pres_score > 0).long() * active).type_as(sent_rep)
        else:
            on_mask = (labels["pres_labels"] * active).type_as(sent_rep)
        on_mask *= active.type_as(sent_rep)
        # predict node value for leaves
        span_score = None
        cat_score = None
        if self.node_type == "internal":
            pass
        elif self.node_type in ["span-single", "span-set"]:
            span_score = self.node_score_val(node_rep, sentence_rep)
        elif self.node_type in ["categorical-single", "categorical-set"]:
            cat_score = self.node_score_val(node_rep)
            if self.node_type == "categorical-single":
                if labels is None:
                    cat_val = cat_score.max(dim=-1)[1]
                else:
                    cat_val = labels["cat_labels"].type_as(sent_mask)
                cat_rep = self.node_val_vec(cat_val)
                node_x = node_x + cat_rep
        else:
            print(self.node_type)
            raise NotImplementedError
        # if using recurrence, update hidden state
        if recursion in ["seq2tree-rec", "sentence-rec"]:
            prev_h, parent_h, _, _ = tree_context
            node_h = self.node_sen_rep._apply_recurrence(node_x, prev_h, parent_h, on_mask)
        elif recursion in ["none", "top-down", "dfs-intern", "dfs-all"]:
            node_h = None
        else:
            raise NotImplementedError
        return {
            "pres_score": pres_score,
            "cat_score": cat_score,
            "span_score": span_score,
            "node_rep": node_rep,
            "node_h": node_h,
        }

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.node_vec = self.node_vec.to(*args, **kwargs)
        self.node_score_pres = self.node_score_pres.to(*args, **kwargs)
        if self.node_score_val is not None:
            self.node_score_val = self.node_score_val.to(*args, **kwargs)
        return self

    # xavier initialization for most parameters
    def initialize(self, mul=1.0):
        torch.nn.init.normal_(self.node_vec, mul / math.sqrt(self.dim))
        self.node_score_pres.initialize(mul)
        if self.node_score_val is not None:
            self.node_score_val.initialize(mul)


##### Define full tree
class ActionTreePrediction(nn.Module):
    def __init__(self, sentence_embed, action_tree_path, args):
        super(ActionTreePrediction, self).__init__()
        self.args = args
        self.dim = args.model_dim
        self.dropout = args.dropout
        action_tree = ActionTree()
        action_tree.from_dict(json.load(open(action_tree_path)))
        self.action_tree = action_tree
        self.root_node = self.action_tree.root
        self.sentence_rep = sentence_embed
        # make tree-wide module parameters
        sent_attn = MultiHeadedAttention(args.sent_heads, self.dim, self.dropout)
        tree_attn = MultiHeadedAttention(args.tree_heads, self.dim, self.dropout)
        rec_cell = nn.GRUCell(self.dim, self.dim)
        self.node_sen_rep = NodeSentenceRep(self, sent_attn, tree_attn, rec_cell)
        self.start_h = nn.Parameter(torch.zeros(self.dim))
        # initialize list of nodes
        self.module_dict = {}
        self.collapse = args.collapse_nodes  # use same parameters for nodes with the same name
        self.node_list = []
        self.make_modules()
        self.module_list = nn.ModuleList([node.predictor for node in self.node_list])
        # index node to easily access representation and mask
        for i, node in enumerate(self.node_list):
            node.tree_index = i + 1
        self.tree_size = len(self.node_list) + 1
        self.context_rep = None
        self.context_mask = None

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.start_h = self.start_h.to(*args, **kwargs)
        self.node_sen_rep = self.node_sen_rep.to(*args, **kwargs)
        for node in self.node_list:
            node.predictor = node.predictor.to(*args, **kwargs)
        return self

    def initialize(self, mul=1.0):
        torch.nn.init.constant_(self.start_h, 0)
        self.node_sen_rep.initialize(mul)
        for node in self.node_list:
            node.predictor.initialize(mul)
        return self

    def device(self):
        return self.start_h.device

    # create Internal-, Categorical-, and Span-Prediction
    # modules to match the structure of self.action_tree
    def make_modules(self):
        self._make_modules_recursion(self.root_node)

    def _make_modules_recursion(self, node):
        # print(len(self.node_list), node.name, node.node_type, node.node_id)
        if self.collapse and node.name in self.module_dict:
            node.predictor = self.module_dict[node.name]
        else:
            node.predictor = NodePrediction(self.node_sen_rep, node, self.args)
            if self.collapse:
                self.module_dict[node.name] = node.predictor
        self.node_list.append(node)
        for k, chld in node.leaf_children.items():
            if self.collapse and chld.name in self.module_dict:
                chld.predictor = self.module_dict[chld.name]
            else:
                chld.predictor = NodePrediction(self.node_sen_rep, chld, self.args)
                if self.collapse:
                    self.module_dict[chld.name] = chld.predictor
            self.node_list.append(chld)
        for k, chld in node.internal_children.items():
            self._make_modules_recursion(chld)

    def _make_tree_context(self, node_h, node_path):
        prev_h, parent_h = node_h
        if node_path:
            c_rep = torch.cat([node.out["node_rep"].unsqueeze(dim=1) for node in node_path], dim=1)
            c_mask = torch.cat([node.mask.unsqueeze(dim=1) for node in node_path], dim=1)
            return (prev_h, parent_h, c_rep, c_mask)
        else:
            return (prev_h, parent_h, None, None)

    # compute internal, categorical and span scores
    # (they have different losses, hence the split)
    # stores scores in the node.out field
    def forward(self, sent_ids, sent_mask, sent_len, recursion="none", prediction=False):
        # compute sentence representation
        sent_rep = self.sentence_rep(sent_ids, sent_mask, sent_len)
        sentence_rep = (sent_rep, sent_mask)
        # initialize root node
        if prediction:
            for node in self.node_list:
                node.active = torch.zeros(sent_rep.shape[0]).type_as(sent_mask)
        self.root_node.active = torch.ones(sent_rep.shape[0]).type_as(sent_mask)
        self.root_node.mask = torch.ones(sent_rep.shape[0]).type_as(sent_mask)
        self.root_node.out = {}
        self.root_node.out["node_rep"] = self.start_h[None, :].expand(sent_rep.shape[0], self.dim)
        self.root_node.labels = {"pres_labels": self.root_node.mask}
        # prepare node path and recurrent state
        node_path = None
        prev_h = None
        parent_h = None
        context_rep = None
        context_mask = None
        if recursion in ["top-down", "dfs-intern", "dfs-all"]:
            node_path = [self.root_node]
        if recursion in ["seq2tree-rec", "sentence-rec"]:
            prev_h = torch.zeros(sent_rep.shape[0], self.dim).type_as(sent_rep)
            # prev_h = prev_h + self.start_h
            parent_h = torch.zeros(sent_rep.shape[0], self.dim).type_as(sent_rep)
        node_h = (prev_h, parent_h)
        # launch recursion
        _ = self._forward_recursion(
            self.root_node, sentence_rep, node_h, node_path, recursion, prediction
        )
        # TODO: add set prediction leaves
        node_scores = [
            node.out for node in self.node_list if any(node.active) and node.name != "root"
        ]
        return node_scores

    # different types of recursion:
    # - 'none': no tree context
    # - 'top-down': only see context from ancestors
    # - 'dfs-intern': sees all internal nodes in DFS order
    # - 'dfs-all': sees all nodes in DFS order
    # - 'seq2tree-rec': Recurrent model following Dong&Lapata
    # - 'sentence-rec': same but recurrent update depends on sentence
    def _forward_recursion(self, node, sentence_rep, node_h, node_path, recursion, prediction):
        prev_h, parent_h = node_h
        # start by predicting presence of current node
        node.out = node.predictor(
            sentence_rep,
            self._make_tree_context(node_h, node_path),
            recursion,
            node.active,
            None if prediction and node.name != "root" else node.labels,
        )
        if recursion == "dfs-all":
            node_path += [node]
        if prediction and node.name != "root":
            node.mask = node.active * (node.out["pres_score"] > 0).type_as(node.active)
        else:
            node.mask = node.active * node.labels["pres_labels"].type_as(node.active)
        if node.node_type == "internal":
            # don't waste time on inactive nodes
            if node.mask.sum().item() == 0:
                node_h = (node.out["node_h"], parent_h)
                return (node_path, node_h)
            # add current node to path for attention-based recursion
            if recursion in ["top-down", "dfs-intern"]:
                node_path += [node]
            # predict the state of all children in DFS order
            node_h = (node.out["node_h"], node.out["node_h"])
            for k, chld in node.children.items():
                chld.active = node.mask
                if recursion in ["dfs-intern", "dfs-all", "seq2tree-rec", "sentence-rec"]:
                    node_path, node_h = self._forward_recursion(
                        chld, sentence_rep, node_h, node_path, recursion, prediction
                    )
                elif recursion in ["none", "top-down"]:
                    _, _ = self._forward_recursion(
                        chld, sentence_rep, node_h, node_path, recursion, prediction
                    )
                else:
                    raise NotImplementedError
            node_h = (node.out["node_h"], parent_h)
        return (node_path, node_h)

    # run through the batch of trees (list of dicts) once to identify nodes that
    # are active in this batch and create labels
    # results are stored in the node.labels and node.active fields
    def make_labels(self, tree_list, cuda=False):
        batch_size = len(tree_list)
        # initialize context_rep and context_mask
        for node in self.node_list:
            node.active = torch.LongTensor([0 for _ in tree_list]).to(self.device())
        self.root_node.active = torch.LongTensor([1 for _ in tree_list]).to(self.device())
        self.root_node.mask = torch.LongTensor([1 for _ in tree_list]).to(self.device())
        self._make_labels_recursion(self.root_node, tree_list, self.root_node.active)
        # FIXME: set nodes
        active_node_list = [
            node
            for node in self.node_list
            if node.active.sum().item() > 0
            and node.name != "root"
            and node.node_type != "span-set"
        ]
        return active_node_list

    def _make_labels_recursion(self, node, sub_tree_list, active):
        # aggregate subtrees
        node.active = active
        node.mask = torch.LongTensor([1 if st else 0 for st in sub_tree_list]).to(self.device())
        cat_labels = None
        span_labels = None
        if node.node_type == "internal":
            pass
        elif node.node_type == "categorical-single":
            cat_labels = torch.LongTensor(
                [node.choices.index(val) if val else 0 for val in sub_tree_list]
            ).to(self.device())
        elif node.node_type == "categorical-set":
            # TODO: FIXME
            # currently predicting one value at random
            cat_labels = torch.LongTensor(
                [
                    node.choices.index(val[0]) if val and len(val) > 0 else 0
                    for val in sub_tree_list
                ]
            ).to(self.device())
        elif node.node_type == "span-single":
            span_labels = torch.LongTensor(
                [(val[0], val[1]) if val else (0, 0) for val in sub_tree_list]
            ).to(self.device())
        node.labels = {
            "pres_labels": node.mask,
            "cat_labels": cat_labels,
            "span_labels": span_labels,
        }
        # check whether we even need to go down and continue
        if node.node_type == "internal" and any(sub_tree_list):
            for k, chld in node.children.items():
                st_list = [t.get(k, False) if t else False for t in sub_tree_list]
                self._make_labels_recursion(chld, st_list, node.mask)

    # predict tree for single-sentence batch
    def predict_tree(self, sent_ids, sent_mask, sent_len, recursion):
        for node in self.node_list:
            node.active = torch.LongTensor([0 for _ in range(sent_ids.shape[0])]).to(self.device())
        self.root_node.active = torch.LongTensor([1 for _ in range(sent_ids.shape[0])]).to(
            self.device()
        )
        self.root_node.mask = torch.LongTensor([1 for _ in range(sent_ids.shape[0])]).to(
            self.device()
        )
        self.root_node.labels = {
            "pres_labels": torch.LongTensor([1 for _ in range(sent_ids.shape[0])]).to(
                self.device()
            )
        }
        _ = self.forward(sent_ids, sent_mask, sent_len, recursion=recursion, prediction=True)
        res = self._predict_tree_recursion(self.root_node)
        return res

    def _predict_tree_recursion(self, node):
        res = {}
        for k, chld in node.children.items():
            if chld.node_type == "internal":
                if chld.mask.item() == 1:
                    res[k] = self._predict_tree_recursion(chld)
            elif chld.node_type == "categorical-single":
                if chld.mask.item() == 1:
                    cat_index = chld.out["cat_score"].max(dim=-1)[1][0].item()
                    res[k] = chld.choices[cat_index]
            elif chld.node_type == "categorical-set":
                # FIXME: currently only / always predicting one
                if chld.mask.item() == 1:
                    cat_index = chld.out["cat_score"].max(dim=-1)[1][0].item()
                    res[k] = [chld.choices[cat_index]]
            elif chld.node_type == "span-single":
                if chld.mask.item() == 1:
                    sp_index = chld.out["span_score"].view(1, -1).max(dim=-1)[1][0].item()
                    start_index = sp_index // chld.out["span_score"].shape[-1]
                    end_index = sp_index % chld.out["span_score"].shape[-1]
                    res[k] = (0, (start_index, end_index))
            else:
                # TODO: add set prediction
                # raise NotImplementedError
                res[k] = "NOT_IMPLEMENTED"
        return res


################# utility
def make_model(args, embeddings_path, action_tree_path, use_cuda):
    map_location = "cpu" if not torch.cuda.is_available() else None
    ft_dict = torch.load(embeddings_path, map_location=map_location)
    i2w = [w for w, c in ft_dict["vocabulary"]]
    w2i = dict([(w, i) for i, w in enumerate(i2w)])
    # make model
    if args.learn_embeds:
        word_embed = FastTextSentenceEmbedding(
            args.model_dim, args.learn_embed_dim, len(i2w), None
        )
    else:
        word_embed = FastTextSentenceEmbedding(
            args.model_dim, args.learn_embed_dim, len(i2w), ft_dict["embeddings"]
        )
    if args.sentence_encoder == "convolution":
        sent_embed = SentenceConvEncoder(
            word_embed, args.model_dim, args.sent_conv_window, args.sent_conv_layers, args.dropout
        )
    elif args.sentence_encoder == "bigru":
        sent_embed = SentenceBiGRUEncoder(
            word_embed, args.model_dim, args.sent_gru_layers, args.dropout
        )
    else:
        raise NotImplementedError
    # load action tree
    # TODO: add initializer multiplier to arguments
    a_tree_pred = ActionTreePrediction(sent_embed, action_tree_path, args)
    a_tree_pred.initialize(mul=1.0)
    a_tree_loss = TreeLoss(args)
    if use_cuda:
        print("moving to cuda")
        a_tree_pred.to("cuda:0")
        a_tree_loss.to("cuda:0")
    return (a_tree_pred, a_tree_loss, w2i)


def load_model(model_path, embeddings_path=None, action_tree_path=None, use_cuda=False, epoch=-1):
    model_name = model_path.split(".pth")[0].strip()
    with open("%s.args.pk" % (model_name,), "rb") as f:
        args = pickle.load(f)
    args.cuda = use_cuda
    args.collapse_nodes = False  # TODO fix
    a_tree_pred, a_tree_loss, w2i = make_model(args, embeddings_path, action_tree_path, use_cuda)
    map_location = None if use_cuda and torch.cuda.is_available() else "cpu"
    if epoch < 0:
        a_tree_pred.load_state_dict(
            torch.load("%s.pth" % (model_name,), map_location=map_location)
        )
        print("loaded %s.pth" % (model_name,))
    else:
        a_tree_pred.load_state_dict(
            torch.load("%s_%d.pth" % (model_name, epoch), map_location=map_location)
        )
        print("loaded %s_%d.pth" % (model_name, epoch))
    a_tree_pred = a_tree_pred.eval()
    return (a_tree_pred, a_tree_loss, w2i, args)
