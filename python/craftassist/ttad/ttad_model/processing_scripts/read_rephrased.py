"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

from copy import deepcopy

# from pprint import pprint
import csv
import json

from spacy.lang.en import English

tokenizer = English().Defaults.create_tokenizer()


def word_tokenize(st):
    return [(x.text, x.idx) for x in tokenizer(st)]


rephrases = []
for j in range(5):
    with open("rephrase_%d.csv" % (j,)) as csvfile:
        g_reader = csv.reader(csvfile)
        for i, row in enumerate(g_reader):
            if i > 0:
                rephrases += [row[-2:]]


brackets = [("(", ")"), ("{", "}"), ("[", "]"), ("*", "*"), ("$", "$"), ("#", "#")]

bracket_chars = ["(", ")", "{", "}", "[", "]", "*", "$", "#"]


def remove_brackets(br_sen):
    br_found = []
    for bs, be in brackets:
        if bs in br_sen:
            idx_s = br_sen.index(bs)
            br_right = br_sen[idx_s + 1 :]
            # idx_s         -= sum([c in bracket_chars for c in br_sen[:idx_s]])
            idx_e = br_right.index(be)
            # idx_e                 -= sum([c in bracket_chars for c in br_sen[:idx_e]])
            br_pre = br_right[:idx_e]
            if False:
                br_str = " ".join([x[0] for x in word_tokenize(br_pre)])
            else:
                br_str = br_pre
            br_found += [(idx_s + 1, idx_s + 1 + idx_e, br_str, bs)]
    no_br = br_sen
    for c in bracket_chars:
        no_br = no_br.replace(c, " ")
    sen_ls = [x for x in word_tokenize(no_br) if x[0].strip() != ""]
    word_spans = []
    for id_s, id_e, br_st, b in br_found:
        idw_s = [n for w, n in sen_ls].index(id_s)
        if sen_ls[-1][1] <= id_e:
            idw_e = len(sen_ls) - 1
        else:
            idw_e = [n > id_e for w, n in sen_ls].index(True) - 1
        word_spans += [(idw_s, idw_e, b, br_st)]
    return (" ".join([w for w, n in sen_ls]), word_spans)


rep_processed = []
for i, (orig, rep) in enumerate(rephrases):
    try:
        rep_processed += [(orig, remove_brackets(orig), remove_brackets(rep))]
    except:
        print("MISSED", i)


# make trees for rephrased data

orig_valid = json.load(open("../bracketted_valid.json"))
rephrases = rep_processed

orig_dict = dict([(x[1], x) for x in orig_valid])

rephrase_trees = []
for i, (orig_br, (orig_nobr, orig_br_lst), (rep_nobr, rep_br_lst)) in enumerate(rephrases):
    try:
        orig_valid_nobr, orig_valid_br, orig_valid_br_lst, orig_valid_tree = orig_dict[orig_br]
        rep_tree = deepcopy(orig_valid_tree)
        list(rep_tree.values())[0]["description"] = rep_nobr
        br_to_span = dict([(b, (id_b, id_e)) for id_b, id_e, b, _ in rep_br_lst])
        for b, (node_path, span) in orig_valid_br_lst:
            node = rep_tree
            tag_path = node_path.split(" > ")
            for k in tag_path[:-1]:
                node = node[k]
            node[tag_path[-1]] = br_to_span[b]
        rephrase_trees += [(rep_nobr, rep_tree)]
    except:
        print("MISSED", i)


json.dump(rephrase_trees, open("../valid_rephrase_02_15.json", "w"))
