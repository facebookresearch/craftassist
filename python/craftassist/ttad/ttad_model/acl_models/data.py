import json
import numpy as np

from random import choice, random, randint, seed, shuffle

import torch

from .action_tree import *


def tree_to_action_type(tr):
    a_type = tr["action_type"]
    if a_type == "Build" and "reference_object" in tr:
        return "Build-Copy"
    else:
        return a_type


def tree_to_nodes(tr, all_nodes=False):
    if "action" in tr:
        return "++".join(
            [tr["action"]["action_type"]] + sorted(set(tree_to_nodes_rec(tr, all_nodes, "")))
        )
    else:
        return "++".join([tr["dialogue_type"]] + sorted(set(tree_to_nodes_rec(tr, all_nodes, ""))))


def tree_to_nodes_rec(tr, all_nodes, ancestor_path):
    res = []
    for k, v in tr.items():
        if type(v) == dict:
            node_path = ancestor_path + "-" + k
            res += [node_path] + tree_to_nodes_rec(v, all_nodes, node_path)
        elif all_nodes:
            node_path = ancestor_path + ":" + k
            res += [node_path]
    return res


# data loader that samples training and validation examples
# according to test distribution
class DataLoader:
    def __init__(self, args):
        seed(1111)
        np.random.seed(1111)
        # load data
        # self.data is a mapping of data split and data type to list of examples
        # e.g. self.data['train']['templated'] = [(description, tree) for _ in range(#train_templated)]
        # it needs to have at least 'train'-'templated' and 'train'-'rephrased' if args.rephrase_proba > 0
        self.data = json.load(open(args.data_file))
        if args.train_on_everything:
            for spl in ["valid", "test"]:
                for data_type in self.data[spl]:
                    self.data["train"]["rephrased"] += self.data[spl][data_type][:]
                    self.data["train"]["templated"] += self.data[spl][data_type][:]
        print("loaded data")
        # organize data so we can resample training set
        self.resample_base = "templated"
        self.resample_mode = args.resample_mode
        self.resample_target = args.resample_type
        if args.resample_mode == "none":
            self.tree_rep = lambda tr: "CONST"
        elif args.resample_mode == "action-type":
            self.tree_rep = lambda tr: tree_to_action_type(tr)
        elif args.resample_mode == "tree-internal":
            self.tree_rep = lambda tr: tree_to_nodes(tr, all_nodes=False)
        elif args.resample_mode == "tree-full":
            self.tree_rep = lambda tr: tree_to_nodes(tr, all_nodes=True)
        else:
            raise NotImplementedError
        self.data_map = {}
        for spl, spl_dict in self.data.items():
            self.data_map[spl] = {}
            for d_type, d_list in spl_dict.items():
                print("mapping", spl, d_type)
                self.data_map[spl][d_type] = {}
                self.data_map[spl][d_type]["<ALL>"] = d_list[:]
                for desc, tr in d_list:
                    trep = self.tree_rep(tr)
                    self.data_map[spl][d_type][trep] = self.data_map[spl][d_type].get(trep, [])
                    self.data_map[spl][d_type][trep] += [(desc, tr)]
        # prepare for sampling without replacement
        all_keys = [
            trep
            for spl, spl_dict in self.data_map.items()
            for d_type, t_reps in spl_dict.items()
            for trep in t_reps
        ]
        all_keys = sorted(set(all_keys))
        base_probas = [
            len(self.data_map["train"][self.resample_base].get(k, []))
            for i, k in enumerate(all_keys)
        ]
        base_probas = [x / sum(base_probas) for x in base_probas]
        target_probas = [
            len(self.data_map["valid"][self.resample_target].get(k, []))
            if base_probas[i] > 0
            else 0
            for i, k in enumerate(all_keys)
        ]
        target_probas = [x / sum(target_probas) for x in target_probas]
        self.sample_keys = all_keys
        self.sample_prob = 0.2 * np.array(base_probas) + 0.8 * np.array(target_probas)
        self.buffer_size = args.hard_buffer_size
        self.buffer_prob = args.hard_buffer_proba
        self.hard_buffer = self.data["train"]["templated"][: self.buffer_size]
        self.rephrase_prob = args.rephrase_proba
        # keep track of which examples have been used
        self.data_log = {}
        for spl, spl_dict in self.data_map.items():
            self.data_log[spl] = {}
            for d_type, trep_to_list in spl_dict.items():
                self.data_log[spl][d_type] = {}
                for trep in trep_to_list:
                    self.data_log[spl][d_type][trep] = 0
        # pre-sample keys
        print("pre-computing samples")
        self.pre_sampled = np.random.choice(self.sample_keys, 128000, p=self.sample_prob)
        self.sample_id = 0
        print("pre-computed samples")

    def next_batch(self, batch_size, mode, src, resample=False):
        batch = []
        for i in range(batch_size):
            if mode == "train":
                if random() < self.buffer_prob:
                    batch += [choice(self.hard_buffer)]
                    continue
                elif random() < self.rephrase_prob:
                    src_choice = "rephrased"
                else:
                    src_choice = "templated"
            else:
                src_choice = src
            if mode == "train" or resample:
                key_choice = self.pre_sampled[self.sample_id]
                self.sample_id = (self.sample_id + 1) % len(self.pre_sampled)
                if key_choice not in self.data_log[mode][src_choice]:
                    key_choice = "<ALL>"
                if self.sample_id == 0:
                    self.pre_sampled = np.random.choice(
                        self.sample_keys, 128000, p=self.sample_prob
                    )
            else:
                key_choice = "<ALL>"
            key_index = self.data_log[mode][src_choice][key_choice]
            batch += [self.data_map[mode][src_choice][key_choice][key_index]]
            # cycle through data and shuffle training data when it runs out
            self.data_log[mode][src_choice][key_choice] += 1
            if self.data_log[mode][src_choice][key_choice] >= len(
                self.data_map[mode][src_choice][key_choice]
            ):
                if mode == "train":
                    shuffle(self.data_map[mode][src_choice][key_choice])
                self.data_log[mode][src_choice][key_choice] = 0
                # print('restarts', src_choice, act_choice)
        return batch

    def update_buffer(self, mistakes):
        for i, mistake in enumerate(mistakes):
            b_idx = randint(0, self.buffer_size - 1)
            self.hard_buffer[b_idx] = mistake

    # always use the same resampled validation set
    def reset_valid(self):
        seed(1111)
        np.random.seed(1111)
        for d_type, trep_map in self.data_log["valid"].items():
            for trep in trep_map:
                self.data_log["valid"][d_type][trep] = 0


# utility functions to concatenate chats
def join_chats(desc_action_list):
    res = []
    for c_ls_ordered, tree in desc_action_list:
        if type(c_ls_ordered) == str:  # for other data sources
            c_list = [c_ls_ordered]
        else:
            # NOTE: this will be deprectaed soon
            c_list = [" ".join(c_ls_ordered[-i - 1].split()[1:]) for i in range(len(c_ls_ordered))]
            # c_list = [" ".join(c_ls_ordered[-i - 1].split()) for i in range(len(c_ls_ordered))]

        index_map = {}
        tot_words = 0
        for c_idx, c in enumerate(c_list):
            for w_idx, w in enumerate(c.split()):
                index_map[(c_idx, w_idx)] = tot_words
                tot_words += 1
            tot_words += 1
        joined_c = " <s> ".join(c_list)
        mapped_tree = map_spans(tree, index_map)

        res += [(joined_c, mapped_tree)]
    return res


def map_spans(tree, index_map):
    for k, v in tree.items():
        if is_span(v):
            l, (s, e) = v
            tree[k] = (index_map[(l, s)], index_map[(l, e)])
        elif is_span_list(v):
            tree[k] = [(index_map[(l, s)], index_map[(l, e)]) for l, (s, e) in v]
        elif is_sub_tree(v):
            tree[k] = map_spans(v, index_map)
        else:
            continue
    return tree


# takes list of (description, action_tree) pairs and makes a batch to send to the model
def make_batch(desc_action_list, w2i, cuda=False, sentence_noise=0.0):
    processed_list = join_chats(desc_action_list)
    sentences = [s.strip() for s, a_tree in processed_list]
    tree_list = [a_tree for s, a_tree in processed_list]
    # sen_tabs = [['<s>'] + s.split() + ['</s>'] for s in sentences]
    sen_tabs = [s.split() for s in sentences]
    max_s_len = max([len(s) for s in sen_tabs])
    sen_tabs_padded = [s + ["<pad>"] * (max_s_len - len(s)) for s in sen_tabs]
    unk_id = w2i["<unk>"]
    # each word is replaced by <unk> with probability sentence_noise
    sent_ids = torch.LongTensor(
        [
            [w2i.get(w, unk_id) if random() >= sentence_noise else unk_id for w in s]
            for s in sen_tabs_padded
        ]
    )
    sent_mask = torch.LongTensor([[0 if w == "<pad>" else 1 for w in s] for s in sen_tabs_padded])
    sent_lengths = torch.LongTensor([len(s) for s in sen_tabs])
    if cuda:
        sent_ids = sent_ids.cuda()
        sent_mask = sent_mask.cuda()
        sent_lengths = sent_lengths.cuda()
    return sent_ids, sent_mask, sent_lengths, tree_list
