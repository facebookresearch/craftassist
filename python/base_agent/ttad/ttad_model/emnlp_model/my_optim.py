"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import logging
from collections import OrderedDict
from time import time
import torch
import torch.nn as nn
import copy
from torch.nn.modules.loss import _Loss

from .data import *


class LabelSmoothingBCE(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothingBCE, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        smooth_target = target.clone().masked_fill(target == 1, self.confidence)
        smooth_target = smooth_target.masked_fill(target == 0, self.smoothing)
        return self.criterion(x, smooth_target)


class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothingCE, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        lprobs = torch.log_softmax(x, dim=-1)
        nll_loss = -lprobs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        smooth_loss = -lprobs.clamp(0, 100).sum(
            dim=-1
        )  # hacky way to deal with sentence position padding
        smooth_val = self.smoothing / x.size(-1)
        loss = (1.0 - self.smoothing) * nll_loss + smooth_val * smooth_loss
        scores = loss
        return scores


class TreeLoss(_Loss):
    def __init__(self, args):
        super(TreeLoss, self).__init__()
        self.label_smoothing = args.label_smoothing
        if args.label_smoothing > 0:
            self.bce_loss = LabelSmoothingBCE(args.label_smoothing)
            self.ce_loss = LabelSmoothingCE(args.label_smoothing)
        else:
            self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
            self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, node_list):
        # go through list of node scores, labels, and per-batch activity
        # to compute loss and accuracy
        # internal nodes: (node_scores, (node_labels, node_active))
        pres_list = []
        span_list = []
        cat_list = []
        for node in node_list:
            pres_scores = node.out["pres_score"]
            pres_labels = node.labels["pres_labels"].type_as(pres_scores)
            node_active = node.active.type_as(pres_scores)
            pres_loss = self.bce_loss(pres_scores, pres_labels) * node_active
            # print('>>>>>>', node.name)
            # print(node.name, pres_scores, pres_labels)
            pres_accu = ((pres_scores > 0) == (pres_labels > 0.5)).type_as(pres_scores)
            # print(pres_accu)
            pres_list += [(pres_loss, pres_accu, node_active)]
            if node.node_type == "internal":
                continue
            elif node.node_type in ["categorical-single", "categorical-set"]:
                # FIXME: currently predicting single value for set
                cat_scores = node.out["cat_score"]
                cat_labels = node.labels["cat_labels"]
                cat_loss = self.ce_loss(cat_scores, cat_labels) * node_active * pres_labels
                cat_accu = (1 - pres_labels) + pres_labels * (
                    cat_scores.argmax(dim=-1) == cat_labels
                ).type_as(pres_scores)
                cat_list += [(cat_loss, cat_accu, node_active)]
            elif node.node_type == "span-single":
                span_pre_scores = node.out["span_score"]  # B x T x T
                span_pre_labels = node.labels["span_labels"]  # B x 2
                span_scores = span_pre_scores.view(span_pre_scores.shape[0], -1)
                span_labels = (
                    span_pre_scores.shape[-1] * span_pre_labels[:, 0] + span_pre_labels[:, 1]
                )
                span_loss = self.ce_loss(span_scores, span_labels) * node_active * pres_labels
                span_accu = (1 - pres_labels) + pres_labels * (
                    span_scores.argmax(dim=-1) == span_labels
                ).type_as(pres_scores)
                span_list += [(span_loss, span_accu, node_active)]
            else:
                # continue
                # TODO fix span-set and categorical-set
                raise NotImplementedError
        # from pprint import pprint
        # pprint(pres_list, width=230)
        pres_loss = torch.cat([l.unsqueeze(1) for l, a, act in pres_list], dim=1)
        pres_loss = pres_loss.sum(dim=1)
        pres_accu = torch.cat(
            [((a * act) == act).unsqueeze(1).type_as(a) for l, a, act in pres_list], dim=1
        )
        pres_accu = pres_accu.sum(dim=1) == len(pres_list)
        # categorical
        cat_loss = torch.cat([l.unsqueeze(1) for l, a, act in cat_list], dim=1)
        cat_loss = cat_loss.sum(dim=1)
        cat_accu = torch.cat(
            [((a * act) == act).unsqueeze(1).type_as(a) for l, a, act in cat_list], dim=1
        )
        cat_accu = cat_accu.sum(dim=1) == len(cat_list)
        # spans
        span_loss = torch.cat([l.unsqueeze(1) for l, a, act in span_list], dim=1)
        span_loss = span_loss.sum(dim=1)
        span_accu = torch.cat(
            [((a * act) == act).unsqueeze(1).type_as(a) for l, a, act in span_list], dim=1
        )
        span_accu = span_accu.sum(dim=1) == len(span_list)
        # aggregate
        res = OrderedDict()
        res["loss"] = span_loss + pres_loss + cat_loss
        res["accuracy"] = (span_accu.long() + pres_accu.long() + cat_accu.long()) == 3
        res["presence_loss"] = pres_loss
        res["categorical_loss"] = cat_loss
        res["span_loss"] = span_loss
        res["presence_accuracy"] = pres_accu
        res["categorical_accuracy"] = cat_accu
        res["span_accuracy"] = span_accu
        # from pprint import pprint
        # pprint(res, width=230)
        return res


# send a tensor or nested lists of tensors to cpu
def to_cpu(obj):
    if type(obj) in [list, tuple]:
        return [to_cpu(o) for o in obj]
    else:
        return obj.detach().cpu()


# runs the model over provided data. trains if optimizer is not None
def run_epoch(data_loader, model, loss, w2i, args, mode="train", data_type="", optimizer=None):
    st_time = time()
    n_batches = args.batches_per_epoch if mode == "train" else 4
    tot_ct = 0
    acc_ct = 0
    global_dct = OrderedDict()
    local_dct = OrderedDict()
    res_list = []
    for i in range(n_batches):
        st_time = time()
        batch_list = data_loader.next_batch(args.batch_size, mode, data_type)
        # print("time for next batch: %r" % (time() - st_time))
        st_time = time()
        batch_list_cp = copy.deepcopy(batch_list)
        # print("time for copy: %r" % (time() - st_time))
        st_time = time()
        # make batch on the deepcopy
        s_ids, s_mask, s_len, t_list = make_batch(
            batch_list_cp,
            w2i,
            args.cuda,
            sentence_noise=args.sentence_noise if mode == "train" else 0.0,
        )
        # print("time for make batch: %r" % (time() - st_time))
        st_time = time()
        # make labels and active mask
        active_nodes = model.make_labels(t_list, args.cuda)
        # print("time for make labels: %r" % (time() - st_time))
        st_time = time()
        # run model
        active_nodes_scores = model(s_ids, s_mask, s_len, recursion=args.recursion)
        # print("time for fwd pass: %r" % (time() - st_time))
        st_time = time()
        # compute loss
        loss_dct = loss(active_nodes)
        # print("time for computing loss: %r" % (time() - st_time))
        st_time = time()
        loss_val = loss_dct["loss"].sum() / loss_dct["loss"].shape[0]
        # print("time for computing loss val: %r" % (time() - st_time))
        st_time = time()
        if optimizer is not None:
            optimizer.zero_grad()
            loss_val.backward(retain_graph=True)
            # print("time for loss backward: %r" % (time() - st_time))
            st_time = time()
            optimizer.step()
            # print("time for opt step: %r" % (time() - st_time))
            st_time = time()
            data_loader.update_buffer(
                [
                    example
                    for j, example in enumerate(batch_list)
                    if loss_dct["accuracy"][j].item() == 0
                ]
            )
            # print("time for update buffer: %r" % (time() - st_time))

        for k, v in loss_dct.items():
            global_dct[k] = global_dct.get(k, 0.0) + v.sum().item()
            local_dct[k] = local_dct.get(k, 0.0) + v.sum().item()
        tot_ct += loss_dct["loss"].shape[0]
        acc_ct += loss_dct["loss"].shape[0]
        # print the stats
        if i % args.print_freq == 0:
            tot_loss = global_dct["loss"]
            tot_accu = global_dct["accuracy"]
            logging.info("---- %d" % (i,))
            print("---- %d" % (i,))
            print(
                "GLOBAL --- loss %.2f --- accu %d of %d --- time %.1f"
                % (global_dct["loss"] / tot_ct, global_dct["accuracy"], tot_ct, time() - st_time)
            )
            print(
                "LOCAL  --- loss %.2f --- accu: full %d of %d -- intern %d -- cat %d -- span %d"
                % (
                    local_dct["loss"] / acc_ct,
                    local_dct["accuracy"],
                    acc_ct,
                    local_dct["presence_accuracy"],
                    local_dct["categorical_accuracy"],
                    local_dct["span_accuracy"],
                )
            )

            acc_ct = 0
            for k in local_dct:
                local_dct[k] = 0.0
    print(
        "EPOCH GLOBAL --- loss %.2f --- accu %d of %d = %.2f --- time %.1f"
        % (
            global_dct["loss"] / tot_ct,
            global_dct["accuracy"],
            tot_ct,
            global_dct["accuracy"] / tot_ct,
            time() - st_time,
        )
    )
    logging.info(
        "EPOCH GLOBAL --- loss %.2f --- accu %d of %d = %.2f --- time %.1f"
        % (
            global_dct["loss"] / tot_ct,
            global_dct["accuracy"],
            tot_ct,
            global_dct["accuracy"] / tot_ct,
            time() - st_time,
        )
    )


# end-to-end prediction
def predict_tree(model, sentence_list, w2i, args):
    s_ids, s_mask, s_len, t_list = make_batch(
        [(sentence_list, {})], w2i, args.cuda, sentence_noise=0.0
    )
    tree = model.predict_tree(s_ids, s_mask, s_len, args.recursion)

    # map span indices back
    if type(sentence_list) == str:
        c_list = [sentence_list]
    else:
        # c_list = [" ".join(sentence_list[-i - 1].split()[1:]) for i in range(len(sentence_list))]
        c_list = [sentence_list[-i - 1] for i in range(len(sentence_list))]
    index_map = {}
    tot_words = 0
    for c_idx, c in enumerate(c_list):
        for w_idx, w in enumerate(c.split()):
            index_map[tot_words] = (c_idx, w_idx)
            tot_words += 1
        tot_words += 1
    tree = reverse_map_spans(tree, index_map)

    return tree


def reverse_map_spans(tree, span_map):
    for k, v in tree.items():
        if is_span(v):
            l, (s, e) = v
            l1, ls = span_map[s] if s in span_map else span_map[s + 1]
            l2, le = span_map[e] if e in span_map else span_map[e - 1]
            if l2 > l1:
                le = ls + 1
            tree[k] = (l1, (ls, le))
        elif is_sub_tree(v):
            tree[k] = reverse_map_spans(v, span_map)
        else:
            continue
    return tree


def tree_equal(ground_truth, prediction, only_internal=False):
    ground_truth.pop("has_attribute", None)
    prediction.pop("has_attribute", None)
    is_eq = all([k in ground_truth for k in prediction]) and all(
        [k in prediction for k in ground_truth]
    )
    if not is_eq:
        return is_eq

    for k, v in ground_truth.items():
        if only_internal:
            if type(v) == dict:  # internal node accuracy
                is_eq = is_eq and tree_equal(v, prediction[k], only_internal)
        else:
            if type(v) == dict:  # internal node accuracy
                is_eq = is_eq and tree_equal(v, prediction[k])
            elif type(v) == str:
                is_eq = is_eq and (v == prediction[k])
            elif type(v) in [list, tuple]:
                if len(v) == 2 and type(v[0]) == int:
                    a, (b, c) = prediction[k]  # prediction
                    # y, z = v
                    x, (y, z) = v  # ground truth
                    is_eq = is_eq and ((a, b, c) == (x, y, z))
    return is_eq


def compute_accuracy(model, examples, w2i, args, only_internal=False):
    predicted = [
        (sentence, ground_truth, predict_tree(model, sentence, w2i, args))
        for sentence, ground_truth in examples
    ]

    num_correct = len(
        [
            sentence
            for sentence, ground_truth, prediction in predicted
            if tree_equal(ground_truth, prediction, only_internal)
        ]
    )

    return num_correct / len(predicted)


def compute_accuracy_per_action_type(model, examples, w2i, args, only_internal=False):
    predicted = [
        (sentence, ground_truth, predict_tree(model, sentence, w2i, args))
        for sentence, ground_truth in examples
    ]

    action_type_stats = {}
    for sentence, ground_truth, prediction in predicted:
        action_type = ground_truth["action_type"]
        # compute accuracy for each action type
        if action_type not in action_type_stats:
            action_type_stats[action_type] = [0.0, 0.0]
        action_type_stats[action_type][1] += 1  # total

        if tree_equal(ground_truth, prediction, only_internal):
            action_type_stats[action_type][0] += 1  # correct

    out = {}
    for key in action_type_stats.keys():
        val = action_type_stats[key][0] / action_type_stats[key][1]
        out[key] = "{0:.3f}".format(val)

    return out


def compute_stats(
    ground_truth,
    prediction,
    total_internal,
    correct_internal,
    total_str,
    correct_str,
    total_span,
    correct_span,
):
    # only by value type
    for k, v in ground_truth.items():
        if type(v) == dict:  # internal node accuracy
            total_internal += 1
            if k in prediction:
                if type(prediction[k]) == dict:
                    # true positive
                    correct_internal += 1
                (
                    total_internal,
                    correct_internal,
                    total_str,
                    correct_str,
                    total_span,
                    correct_span,
                ) = compute_stats(
                    v,
                    prediction[k],
                    total_internal,
                    correct_internal,
                    total_str,
                    correct_str,
                    total_span,
                    correct_span,
                )
            else:
                (
                    total_internal,
                    correct_internal,
                    total_str,
                    correct_str,
                    total_span,
                    correct_span,
                ) = compute_stats(
                    v,
                    {},
                    total_internal,
                    correct_internal,
                    total_str,
                    correct_str,
                    total_span,
                    correct_span,
                )
        elif type(v) == str:
            total_str += 1
            if k not in prediction:
                continue
            if type(prediction[k]) == str:
                correct_str += 1
        elif type(v) in [list, tuple]:
            total_span += 1
            if k not in prediction:
                continue
            if type(prediction[k]) in [list, tuple]:
                correct_span += 1
    return total_internal, correct_internal, total_str, correct_str, total_span, correct_span


def compute_precision_recall(model, examples, w2i, args):
    predicted = [
        (sentence, ground_truth, predict_tree(model, sentence, w2i, args))
        for sentence, ground_truth in examples
    ]

    final_stats = {}
    final_stats["intern"] = [0.0, 0.0, 0.0]
    final_stats["str"] = [0.0, 0.0, 0.0]
    final_stats["span"] = [0.0, 0.0, 0.0]

    micro_stats_total = {}
    micro_stats_total["intern"] = [0.0, 0.0, 0.0]
    micro_stats_total["str"] = [0.0, 0.0, 0.0]
    micro_stats_total["span"] = [0.0, 0.0, 0.0]

    for sentence, ground_truth, prediction in predicted:  # each example
        (
            total_intern_recall,
            tp_intern_recall,
            total_str_recall,
            tp_str_recall,
            total_span_recall,
            tp_span_recall,
        ) = compute_stats(ground_truth, prediction, 0, 0, 0, 0, 0, 0)
        (
            total_intern_prec,
            tp_intern_prec,
            total_str_prec,
            tp_str_prec,
            total_span_prec,
            tp_span_prec,
        ) = compute_stats(prediction, ground_truth, 0, 0, 0, 0, 0, 0)
        stats_map = {}
        stats_map["intern"] = [
            tp_intern_prec,
            total_intern_prec,
            tp_intern_recall,
            total_intern_recall,
        ]
        stats_map["str"] = [tp_str_prec, total_str_prec, tp_str_recall, total_str_recall]
        stats_map["span"] = [tp_span_prec, total_span_prec, tp_span_recall, total_span_recall]

        # compute prec, recall and f1
        vals = [0, 0, 0]
        for key, val in stats_map.items():
            # sum up tp, fp and fn for each key
            tp = val[0]  # or val[2]
            fp = val[1] - val[0]  # total_prec - tp
            fn = val[3] - val[0]  # total_recall - tp
            micro_stats_total[key] = [x + y for x, y in zip(micro_stats_total[key], [tp, fp, fn])]

            # now compute macro stats
            if val[0] == 0 and val[1] == 0 and val[3] == 0:
                vals = [1.0, 1.0, 1.0]
            elif val[0] == 0 and (val[1] > 0 or val[3] > 0):
                vals = [0.0, 0.0, 0.0]
            else:
                precision = val[0] / val[1]
                recall = val[2] / val[3]
                f1 = (2 * precision * recall) / (precision + recall)
                vals = [precision, recall, f1]
            final_stats[key] = [x + y for x, y in zip(final_stats[key], vals)]

    macro_stats = {
        key: ["{0:.3f}".format(item / len(examples)) for item in val]
        for key, val in final_stats.items()
    }

    # compute micro stats
    micro_stats = {}
    for key, val in micro_stats_total.items():
        tp, fp, fn = val
        if tp == 0 and fp == 0 and fn == 0:
            vals = [1.0, 1.0, 1.0]
        elif tp == 0.0 and (fp > 0 or fn > 0):
            vals = [0.0, 0.0, 0.0]
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = (2 * precision * recall) / (precision + recall)
            vals = [precision, recall, f1]
        micro_stats[key] = ["{0:.3f}".format(x) for x in vals]

    return macro_stats, micro_stats
