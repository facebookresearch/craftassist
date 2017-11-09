import numpy as np
import argparse
import logging
import random
import torch
import torch.nn as nn
import ttad_data
import ttad_models

from generate_actions import generate_actions

import os
from pprint import pprint

# Set the seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s")
logging.getLogger().setLevel(logging.DEBUG)


class ActionDictBuilder(object):
    def __init__(self, model_path, threshold=0.0):
        self.model = ttad_models.TTADNet(None, path=model_path)
        self.threshold = threshold
        self.word_dict = self.model.word_dict
        self.key_dict = self.model.key_dict
        self.intent_dict = self.model.intent_dict
        self.opts = self.model.opts
        self.debug_mode = False
        self.max_sentence_length = self.opts.max_sentence_length

    def listen(self, chat):
        context = []
        curr_depth = 0
        # TODO do better here, use whatever other tokenizer we are using
        words = chat.strip('"').split()
        tree = {}
        max_depth_exceeded = self.write_tree_recursive(tree, context, words, curr_depth)
        if max_depth_exceeded:
            return {"Noop": {}}
        return tree

    # this recursively applies the model to the growing tree.
    # the "context" is a list giving the direct anscestors of the node
    # the words are the result of splitting the original request
    # note that each node in the tree is either a key pointing to a
    # subtree (represented as a dict) or a leaf (a string)
    def write_tree_recursive(self, tree, context, words, curr_depth):
        curr_depth += 1
        if curr_depth > 10:
            return True

        c = torch.LongTensor(1, self.opts.max_depth)
        c[:] = self.opts.null_key
        for i in range(len(context)):
            c[0][i] = self.key_dict["all_keys"]["w2i"][context[i]]
        w = torch.LongTensor(1, self.opts.max_sentence_length)
        w[:] = self.opts.null_word
        for i in range(len(words)):
            word_idx = self.word_dict["w2i"].get(words[i], ttad_data.UNKWORD)
            w[0][i] = word_idx

        self.model.eval()
        with torch.no_grad():
            key, val, intent, span = self.model.forward(w, c, context_len=len(context))

        if self.debug_mode:
            import pdb

            pdb.set_trace()

        if len(context) == 0:
            val = self.intent_dict["i2w"][torch.max(intent, 1)[1].item()]
            context = [val]
            max_depth_exceeded = self.write_tree_recursive(tree, context, words, curr_depth)
            if max_depth_exceeded:
                return True
            return False

        # this key opens a dict; populate the dict:
        if self.key_dict["dict_open_keys"].get(context[-1]) is not None:
            pidx = torch.nonzero(key > self.threshold)
            child_tree = {}
            tree[context[-1]] = child_tree
            for i in pidx:
                context_child = context.copy()
                child_action = self.key_dict["all_keys"]["i2w"][i[1].item()]
                context_child.append(child_action)
                max_depth_exceeded = self.write_tree_recursive(
                    child_tree, context_child, words, curr_depth
                )
                if max_depth_exceeded:
                    return True
            return False
        # this is a span, use the span head
        elif self.key_dict["span_keys"].get(context[-1]) is not None:
            _, start, end = max(
                [
                    (x + y, L, R)
                    for (L, x) in enumerate(span[0].squeeze().numpy())
                    for (R, y) in enumerate(span[1].squeeze().numpy())
                    if L <= R
                ]
            )  # mask out invalid spans
            tree[context[-1]] = [start, end]
            return False
        # this key just has a str value
        else:
            val = torch.argmax(val.squeeze()).item()
            val = self.key_dict["str_vals"]["i2w"][val]
            tree[context[-1]] = val
            return False

    def annotate(self, chat):
        context = []
        curr_depth = 0
        # TODO do better here, use whatever other tokenizer we are using
        words = chat.strip('"').split()
        tree = {}
        self.write_tree_recursive(tree, context, words, curr_depth)
        # check whether full output looks good
        os.system("clear")
        print(chat, "\n----------")
        pprint(tree)
        response = input("Does this look right? [y/n]\n")
        if response.strip().lower() == "n":
            context = []
            curr_depth = 0
            words = chat.strip('"').split()
            tree = {}
            self.write_tree_recursive_interactive(tree, context, words, curr_depth, chat)
            return (False, tree)
        return (True, tree)

    # same as write_tree_recursive, but checks with user whether
    # the model predictions are accurate
    def write_tree_recursive_interactive(self, tree, context, words, curr_depth, chat=""):
        curr_depth += 1
        if curr_depth > 10:
            return True

        c = torch.LongTensor(1, self.opts.max_depth)
        c[:] = self.opts.null_key
        for i in range(len(context)):
            c[0][i] = self.key_dict["all_keys"]["w2i"][context[i]]
        w = torch.LongTensor(1, self.opts.max_sentence_length)
        w[:] = self.opts.null_word
        for i in range(len(words)):
            word_idx = self.word_dict["w2i"].get(words[i], ttad_data.UNKWORD)
            w[0][i] = word_idx

        self.model.eval()
        with torch.no_grad():
            key, val, intent, span = self.model.forward(w, c, context_len=len(context))

        if self.debug_mode:
            import pdb

            pdb.set_trace()

        if len(context) == 0:
            val = self.intent_dict["i2w"][torch.max(intent, 1)[1].item()]
            st = chat + "\n----------\n"
            st += "Is this a < %s > command? [y/n]\n" % (val,)
            os.system("clear")
            response = input(st)
            if response.strip().lower() == "n":
                st = chat + "\n----------\n"
                st += "Which command is this? [0-%d]\n" % (len(self.intent_dict["i2w"]),)
                for i in range(len(self.intent_dict["i2w"])):
                    st += "%2d:   %s\n" % (i, self.intent_dict["i2w"][i])
                os.system("clear")
                response = input(st + "\n")
                val = self.intent_dict["i2w"][int(response)]
            context = [val]
            max_depth_exceeded = self.write_tree_recursive_interactive(
                tree, context, words, curr_depth, chat
            )
            if max_depth_exceeded:
                return True
            return False

        # this key opens a dict; populate the dict:
        if self.key_dict["dict_open_keys"].get(context[-1]) is not None:
            pidx = torch.nonzero(key > self.threshold)
            child_tree = {}
            tree[context[-1]] = child_tree
            attrs = [self.key_dict["all_keys"]["i2w"][i[1].item()] for i in pidx]
            st = chat + "\n----------\n"
            st += (
                "Is the following list of < %s > arguments accurate and complete: \t [%s] ? [y/n]"
                % (context[-1], ", ".join(attrs))
            )
            os.system("clear")
            response = input(st + "\n")
            if response.strip().lower() == "n":
                st = chat + "\n----------\n"
                st += "Write the space separated list of < %s > arguments mentioned above:\n\n" % (
                    context[-1],
                )
                for i in range(len(self.key_dict["all_keys"]["i2w"])):
                    st += "%2d:   %s\n" % (i, self.key_dict["all_keys"]["i2w"][i])
                os.system("clear")
                response = input(st + "\n")
                attr_idx_list = [int(x) for x in response.strip().split()]
                attrs = [self.key_dict["all_keys"]["i2w"][a] for a in attr_idx_list]
            for attr in attrs:
                context_child = context.copy()
                child_action = attr
                context_child.append(child_action)
                max_depth_exceeded = self.write_tree_recursive_interactive(
                    child_tree, context_child, words, curr_depth, chat
                )
                if max_depth_exceeded:
                    return True
            return False
        # this is a span, use the span head
        elif self.key_dict["span_keys"].get(context[-1]) is not None:
            start = torch.argmax(span[0].squeeze()).item()
            end = torch.argmax(span[1].squeeze()).item()
            st = chat + "\n----------\n"
            st += "Is the value of < %s > described by (%s) ? [y/n]" % (
                context[-1],
                " ".join(words[start : end + 1]),
            )
            os.system("clear")
            response = input(st + "\n")
            if response.strip().lower() == "n":
                st = (
                    " ".join(["%d-%s" % (i, wrd) for i, wrd in enumerate(words)])
                    + "\n----------\n"
                )
                st += "What is the span of < %s >? [start end]" % (context[-1],)
                response = input(st)
                resp_tab = response.strip().split()
                start, end = [int(x) for x in resp_tab]
            tree[context[-1]] = [start, end]
            return False
        # this key just has a str value
        else:
            val = torch.argmax(val.squeeze()).item()
            val = self.key_dict["str_vals"]["i2w"][val]
            st = chat + "\n----------\n"
            st += "Is the value of < %s > (%s) ? [y/n]" % (context[-1], val)
            response = input(st)
            if response.strip().lower() == "n":
                st = chat + "\n----------\n"
                st += "What is the value? [0-%d]" % (len(self.key_dict["str_vals"]["i2w"]),)
                for i in range(len(self.intent_dict["i2w"])):
                    st += "%2d:   %s\n" % (i, self.key_dict["str_vals"]["i2w"][i])
                response = input(st)
                val = self.key_dict["str_vals"]["i2w"][int(response)]
            tree[context[-1]] = val
            return False


def train_epoch(DL, hard_examples, model, optimizers, epoch_number, opts):
    counts = {}
    errors = {}
    data_kinds = ["key", "str", "intent", "span"]
    for k in data_kinds:
        counts[k] = 0
        errors[k] = 0

    hard_examples_added = 0

    if opts.margin >= 0:
        keyloss = ttad_models.MarginHinge(opts.margin)
    else:
        keyloss = nn.BCEWithLogitsLoss(reduce=False)
    loss_dict = {
        "key": keyloss,
        "str": nn.NLLLoss(reduce=False),
        "intent": nn.NLLLoss(reduce=False),
        "span": nn.NLLLoss(reduce=False),
    }
    for batch in DL:
        model.train()

        for part, net in optimizers.items():
            net.zero_grad()

        hard = False
        if torch.rand(1).item() < opts.hard_example_batch_prob:
            if hard_examples.count["key"] + hard_examples.count["intent"] > 20000:
                hard = True
                batch = hard_examples.sample(opts.batchsize * 5)

        for i in data_kinds:
            counts[i] += 1
            if not hard:
                KC = batch[i]["contexts"]
                KW = batch[i]["words"]
                Klen = batch[i]["context_len"]
                gt = batch[i]["gt"]
                if opts.cuda:
                    KC = KC.cuda()
                    KW = KW.cuda()
                    Klen = Klen.cuda()
                    gt = gt.cuda()
                output = model.forward(KW, KC, mode=i, context_len=Klen)
                if i == "span":
                    point_loss = (
                        loss_dict[i](output[0], gt[:, 0]) + loss_dict[i](output[1], gt[:, 1])
                    ) / 2
                else:
                    point_loss = loss_dict[i](output, gt)
                if not hard:
                    max_loss = point_loss.max(dim=-1)[0]
                    big_loss_idx = torch.nonzero(max_loss > opts.hard_example_threshold)
                    for index in big_loss_idx:
                        hard_examples_added += 1
                        j = index.item()
                        try:
                            hard_examples.add_example(KC[j], Klen[j], KW[j], gt[j], max_loss[j], i)
                        except:
                            hard_examples.add_example(
                                KC[j], Klen[j], KW[j], gt[j], max_loss.item(), i
                            )
                mean_loss = point_loss.mean()
                mean_loss.backward()
                errors[i] += mean_loss.item()

        for part, o in optimizers.items():
            o.step()
        if counts["key"] % opts.verbose == 1:
            e = []
            for i in range(len(data_kinds)):
                e.append(errors[data_kinds[i]] / (counts[data_kinds[i]] + 0.0001))
            logging.info(
                "epoch {0:3d} key loss {1:.4f}  str_value loss {2:.4f} intent loss {3:.4f} span loss {4:.4f} \
                    hard examples added  {5:8d}".format(
                    epoch_number, e[0], e[1], e[2], e[3], hard_examples_added
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=20, help="number of epochs")
    parser.add_argument("--cuda", type=int, default=1, help="use cuda if 1")
    parser.add_argument("--batchsize", type=int, default=256, help="batchsize")
    parser.add_argument(
        "--max_sentence_length", type=int, default=25, help="number of words in an input"
    )
    parser.add_argument(
        "--jitter_with_unk_prob",
        type=float,
        default=0.1,
        help="how often (per chat) to put an UNK in at random",
    )
    parser.add_argument("--embedding_dim", type=int, default=64, help="embedding dimension")
    parser.add_argument("--head_dim", type=int, default=32, help="linear head dimension")
    parser.add_argument("--span_head_dim", type=int, default=64, help="span head dimension")
    parser.add_argument("--verbose", type=int, default=100, help="steps before printing errors")
    parser.add_argument(
        "--recurrent_context", type=int, default=-1, help="if > 0 use gru with that many layers"
    )
    parser.add_argument("--conv_body_layers", type=int, default=3, help="layers in body")
    parser.add_argument("--use_batchnorm", type=int, default=1, help="if 1 uses batchnorm")
    parser.add_argument("--nonlin", default="elu", help="relu or elu")
    parser.add_argument("--optimizer", default="adagrad", help="adagrad, adam, or sgd")
    parser.add_argument(
        "--margin",
        type=float,
        default=0.2,
        help="if >= 0 use hinge margin with this parameter instead of bce",
    )
    parser.add_argument("--wd", type=float, default=0.0, help="weight decay")
    parser.add_argument("--all_lr", type=float, default=-0.01, help="learning rate")
    parser.add_argument("--lr_body", type=float, default=0.01, help="learning rate")
    parser.add_argument("--lr_context", type=float, default=0.01, help="learning rate")
    parser.add_argument("--lr_span_head", type=float, default=0.01, help="learning rate")
    parser.add_argument("--lr_intent_head", type=float, default=0.01, help="learning rate")
    parser.add_argument("--lr_str_head", type=float, default=0.01, help="learning rate")
    parser.add_argument("--lr_key_head", type=float, default=0.01, help="learning rate")
    parser.add_argument(
        "--hard_example_batch_prob",
        type=float,
        default=0.25,
        help="chance of sampling a previously hard batch",
    )
    parser.add_argument(
        "--hard_example_threshold", type=float, default=0.1, help="loss to include"
    )
    parser.add_argument(
        "--num_hard_examples", type=int, default=100000, help="number of hard examples in storage"
    )
    parser.add_argument("--mom", type=float, default=0.0, help="momentum")
    parser.add_argument(
        "--load_model", default="", help="from where to load model, empty for no save"
    )
    parser.add_argument(
        "--save_model", default="../models/ttad.pth", help="where to save model, empty for no save"
    )
    parser.add_argument("--epoch_size", type=int, default=100000, help="examples per epoch")
    parser.add_argument("--ndonkeys", type=int, default=8, help="number of dataloader workers")
    opts = parser.parse_args()
    if opts.all_lr > 0:
        opts.lr_body = opts.all_lr
        opts.lr_context = opts.all_lr
        opts.lr_span_head = opts.all_lr
        opts.lr_node_head = opts.all_lr
        opts.lr_intent_head = opts.all_lr

    opts.max_depth = 20  # FIXME get this from the data
    logging.info("getting some data to build dictionary")
    chats, trees = generate_actions(100000)
    word_dict, key_dict, intent_dict = ttad_data.build_dictionaries(chats, trees)
    dictionaries = {}
    dictionaries["word_dict"] = word_dict
    dictionaries["key_dict"] = key_dict
    dictionaries["intent_dict"] = intent_dict

    opts.n_words = len(word_dict["i2w"]) + 1
    opts.n_keys = len(key_dict["i2w"]) + 1
    opts.n_target_keys = len(key_dict["all_keys"]["i2w"])
    opts.null_word = len(word_dict["i2w"])
    opts.null_key = len(key_dict["i2w"])

    hard_examples = ttad_data.HardExampleStore(opts)

    cuda = opts.cuda == 1
    opts.cuda = cuda
    num_epochs = opts.num_epochs
    if opts.load_model != "":
        logging.info("loading model from {}".format(opts.load_model))
        model = ttad_models.TTADNet(opts, path=opts.load_model)
        opts = model.opts
        # todo allow selective override, for now just cuda and num_epochs
        opts.cuda = cuda
        logging.info("overriding opts with saved: {}".format(opts))
    else:
        model = ttad_models.TTADNet(opts, dictionaries=dictionaries)

    if opts.cuda:
        model.cuda()
    optimizers = ttad_models.build_optimizers(opts, model)

    data = ttad_data.ActionDataset(opts, word_dict, key_dict, intent_dict)
    DL = torch.utils.data.DataLoader(
        data,
        batch_size=opts.batchsize,
        pin_memory=True,
        drop_last=False,
        collate_fn=ttad_data.batch_collater,
        num_workers=opts.ndonkeys,
    )

    for i in range(num_epochs):
        train_epoch(DL, hard_examples, model, optimizers, i, opts)
        if opts.save_model != "":
            model.save(opts.save_model)
