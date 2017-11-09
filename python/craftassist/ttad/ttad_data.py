import numpy as np
import logging
import argparse
import torch
import torch.utils.data as dset

# from tqdm import tqdm

from generation.generate_actions import generate_actions

UNKWORD = 0

# TODO extra toplevel action '? don't know '

# add the vocab from a particular tree to the dictionaries
def get_vocab_from_tree(t, dictionary):
    if type(t) is str:
        if dictionary["w2i"].get(t) is None:
            i = len(dictionary["w2i"])
            dictionary["w2i"][t] = i
            dictionary["i2w"][i] = t
            j = len(dictionary["str_vals"]["i2w"])
            dictionary["str_vals"]["i2w"][j] = t
            dictionary["str_vals"]["w2i"][t] = j
    elif type(t) is dict:
        t.pop("description", None)
        for k, v in t.items():
            if type(k) is str:
                if dictionary["w2i"].get(k) is None:
                    i = len(dictionary["w2i"])
                    dictionary["w2i"][k] = i
                    dictionary["i2w"][i] = k
                    if dictionary["all_keys"]["w2i"].get(k) is None:
                        j = len(dictionary["all_keys"]["i2w"])
                        dictionary["all_keys"]["i2w"][j] = k
                        dictionary["all_keys"]["w2i"][k] = j
                    if type(v) is str:
                        dictionary["str_keys"][k] = i
                    elif type(v) is dict:
                        dictionary["dict_open_keys"][k] = i
                    elif type(v) is tuple or type(v) is list:
                        dictionary["span_keys"][k] = i
            get_vocab_from_tree(v, dictionary)


# runs over the data, grabs all vocab
def build_dictionaries(chats, trees):
    logging.info("getting vocab")
    word_dict = {"w2i": {"UNK": UNKWORD}, "i2w": {UNKWORD: "UNK"}}
    intent_dict = {
        "w2i": {
            "Move": 0,
            "Build": 1,
            "Destroy": 2,
            "Tag": 3,
            "Noop": 4,
            "Stop": 5,
            "Resume": 6,
            "Dig": 7,
            "Undo": 8,
            "Fill": 9,
            "Spawn": 10,
            "FreeBuild": 11,
            "Answer": 12,
        },
        "i2w": {
            0: "Move",
            1: "Build",
            2: "Destroy",
            3: "Tag",
            4: "Noop",
            5: "Stop",
            6: "Resume",
            7: "Dig",
            8: "Undo",
            9: "Fill",
            10: "Spawn",
            11: "FreeBuild",
            12: "Answer",
        },
    }
    action_dict = {
        "w2i": {"span": 0},
        "i2w": {0: "span"},
        "str_keys": {},  # keys in the tree whose value is a str
        "str_vals": {"i2w": {}, "w2i": {}},  # string values
        "dict_open_keys": {},  # keys in the tree whose value is a dict
        "span_keys": {},  # keys that open a span
        "all_keys": {"i2w": {}, "w2i": {}},
    }

    for chat in chats:
        words = chat.strip('"').split()
        for w in words:
            if word_dict["w2i"].get(w) is None:
                i = len(word_dict["w2i"])
                word_dict["w2i"][w] = i
                word_dict["i2w"][i] = w
    for tree in trees:
        get_vocab_from_tree(tree, action_dict)
    return word_dict, action_dict, intent_dict


# TODO!!!! more refined unk in data


def read_data(path):
    with open(path) as f:
        L = f.readlines()
        chats = []
        trees = []
        for i in range(len(L) // 3):
            chats.append(L[3 * i].strip())
            trees.append(eval(L[3 * i + 1]))
    return chats, trees


# this takes an action tree and writes single paths down the
# tree into tensors


class TreeEncoder(object):
    def __init__(self, intent_dict, key_dict, opts):
        self.opts = opts
        self.cdict = key_dict["all_keys"]["w2i"]
        self.kdict = key_dict["all_keys"]["w2i"]
        self.sdict = key_dict["str_vals"]["w2i"]
        self.idict = intent_dict["w2i"]

    def write_tensor_lists(self, tree):
        input_context = []
        contexts = {"str": [], "span": [], "key": [], "strlen": [], "spanlen": [], "keylen": []}
        gt = {"str": [], "span": [], "key": []}
        self.write_tensors_recursive(tree, contexts, gt, input_context)
        return contexts, gt

    def write_tensors_recursive(self, tree, contexts, gt, input_context):
        # contexts is a dict, each value is a list of output tensors
        v = torch.LongTensor(self.opts.max_depth)
        v[:] = self.opts.null_key
        input_context_len = len(input_context)
        for i in range(input_context_len):  # TODO write from the other side?
            v[i] = self.cdict[input_context[i]]

        if type(tree) is list:  # this is a span output
            # check to see that the end of the span is not outside max_sentence_length:
            if tree[1] < self.opts.max_sentence_length:
                contexts["span"].append(v)
                contexts["spanlen"].append(input_context_len)
                span_tensor = torch.LongTensor(2)
                span_tensor[0] = tree[0]
                span_tensor[1] = tree[1]
                gt["span"].append(span_tensor)

        elif type(tree) is str:  # this is a named value
            contexts["str"].append(v)
            contexts["strlen"].append(input_context_len)
            gt["str"].append(self.sdict[tree])

        else:  # this is a subtree
            contexts["key"].append(v)
            contexts["keylen"].append(input_context_len)
            if input_context_len == 0:  # an intent, single key for softmax
                gt["key"].append(self.idict[list(tree)[0]])
            else:  # encode all the keys, multilabel problem
                g = torch.Tensor(len(self.kdict)).zero_()
                tree.pop("description", None)
                g[[self.kdict[k] for k in tree.keys()]] = 1
                gt["key"].append(g)
            for k, v in tree.items():
                input_context_child = input_context.copy()
                input_context_child.append(k)
                self.write_tensors_recursive(v, contexts, gt, input_context_child)


def encode_unkify_concat(chat, word_dict, N, opts):
    v = torch.LongTensor(opts.max_sentence_length)
    v[:] = opts.null_word
    chat = chat.strip('"').split()
    chat_length = min(opts.max_sentence_length, len(chat))
    for j in range(chat_length):
        if chat[j] in word_dict["w2i"]:
            v[j] = word_dict["w2i"][chat[j]]
        else:
            v[j] = UNKWORD
    encoded_chats = []
    for i in range(N):
        noisy_v = v.clone()
        if torch.rand(1).item() < opts.jitter_with_unk_prob:
            j = int(torch.randint(chat_length, (1,)).item())
            noisy_v[j] = UNKWORD
        encoded_chats.append(noisy_v)
    return encoded_chats


class ActionDataset(dset.Dataset):
    def __init__(self, opts, word_dict, action_dict, intent_dict):
        super(ActionDataset, self).__init__()
        self.word_dict = word_dict
        self.opts = opts
        self.tree_encoder = TreeEncoder(intent_dict, action_dict, opts)

    def __getitem__(self, i):
        chat, tree = generate_actions(1)
        chat = chat[0]
        tree = tree[0]
        contexts, gt = self.tree_encoder.write_tensor_lists(tree)
        encoded_chat_key = encode_unkify_concat(
            chat, self.word_dict, len(contexts["key"]), self.opts
        )
        encoded_chat_span = encode_unkify_concat(
            chat, self.word_dict, len(contexts["span"]), self.opts
        )
        encoded_chat_str = encode_unkify_concat(
            chat, self.word_dict, len(contexts["str"]), self.opts
        )
        out = {}
        out["intent"] = {
            "context": contexts["key"][0],
            "words": encoded_chat_key[0],
            "gt": gt["key"][0],
        }
        out["key"] = {
            "context": contexts["key"][1:],
            "context_len": contexts["keylen"][1:],
            "words": encoded_chat_key[1:],
            "gt": gt["key"][1:],
        }
        out["str"] = {
            "context": contexts["str"],
            "context_len": contexts["strlen"],
            "words": encoded_chat_str,
            "gt": gt["str"],
        }
        out["span"] = {
            "context": contexts["span"],
            "context_len": contexts["spanlen"],
            "words": encoded_chat_span,
            "gt": gt["span"],
        }
        return out

    def __len__(self):
        return self.opts.epoch_size


def batch_collater(batch):
    out = {}

    B = [b["intent"] for b in batch]
    out["intent"] = {}
    out["intent"]["contexts"] = torch.stack([b["context"] for b in B])
    out["intent"]["context_len"] = torch.cat([torch.LongTensor([0]) for b in B])
    out["intent"]["words"] = torch.stack([b["words"] for b in B])
    out["intent"]["gt"] = torch.LongTensor([b["gt"] for b in B])

    B = [b["str"] for b in batch]
    out["str"] = {}
    try:
        out["str"]["contexts"] = torch.cat(
            [torch.stack(b["context"]) for b in B if len(b["context"]) > 0]
        )
        out["str"]["context_len"] = torch.cat(
            [torch.LongTensor(b["context_len"]) for b in B if len(b["context"]) > 0]
        )
        out["str"]["words"] = torch.cat(
            [torch.stack(b["words"]) for b in B if len(b["context"]) > 0]
        )
        out["str"]["gt"] = torch.cat(
            [torch.LongTensor(b["gt"]) for b in B if len(b["context"]) > 0]
        )
    except:
        import pdb

        pdb.set_trace()

    for part in ["key", "span"]:
        B = [b[part] for b in batch]
        out[part] = {}
        out[part]["contexts"] = torch.cat(
            [torch.stack(b["context"]) for b in B if len(b["context"]) > 0]
        )
        out[part]["context_len"] = torch.cat(
            [torch.LongTensor(b["context_len"]) for b in B if len(b["context"]) > 0]
        )
        out[part]["words"] = torch.cat(
            [torch.stack(b["words"]) for b in B if len(b["context"]) > 0]
        )
        out[part]["gt"] = torch.cat([torch.stack(b["gt"]) for b in B if len(b["context"]) > 0])

    return out


class HardExampleStore(object):
    def __init__(self, opts):
        self.num_examples = opts.num_hard_examples
        data_types = ["key", "intent", "str", "span"]
        self.n_keys = opts.n_keys
        self.max_depth = opts.max_depth
        self.max_sentence_length = opts.max_sentence_length
        N = self.num_examples
        self.count = {t: 0 for t in data_types}
        self.data = {}
        for t in data_types:
            self.data[t] = {}
            self.data[t]["contexts"] = torch.LongTensor(N, opts.max_depth)  # these are all null
            self.data[t]["words"] = torch.LongTensor(N, opts.max_sentence_length)
            self.data[t]["context_len"] = torch.LongTensor(N)
            self.data[t]["scores"] = torch.Tensor(N)

        self.data["intent"]["gt"] = torch.LongTensor(N)
        self.data["key"]["gt"] = torch.Tensor(N, opts.n_target_keys)
        self.data["str"]["gt"] = torch.LongTensor(N)
        self.data["span"]["gt"] = torch.LongTensor(N, 2)

    def add_example(self, context, context_len, words, gt, score, example_type):
        if self.count[example_type] < self.num_examples:
            i = self.count[example_type]
            self.count[example_type] += 1
        else:
            i = np.random.randint(self.num_examples)
        self.data[example_type]["contexts"][i] = context
        self.data[example_type]["context_len"][i] = context_len
        self.data[example_type]["words"][i] = words
        self.data[example_type]["gt"][i] = gt
        self.data[example_type]["scores"][i] = score

    def sample(self, N):
        out = {}
        for i in ["key", "intent", "str", "span"]:
            out[i] = {}
            idx = torch.LongTensor(np.random.randint(self.count[i], size=N))
            out[i]["contexts"] = self.data[i]["contexts"][idx]
            out[i]["context_len"] = self.data[i]["context_len"][idx]
            out[i]["words"] = self.data[i]["words"][idx]
            out[i]["gt"] = self.data[i]["gt"][idx]
        return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_sentence_length", type=int, default=25, help="number of words in an input"
    )
    parser.add_argument(
        "--jitter_with_unk_prob",
        type=float,
        default=0.1,
        help="how often (per chat) to put an UNK in at random",
    )
    parser.add_argument("--epoch_size", type=int, default=500000, help="examples per epoch")
    parser.add_argument("--batchsize", type=int, default=256, help="examples per epoch")
    parser.add_argument("--ndonkeys", type=int, default=0, help="number of dataloader workers")
    opts = parser.parse_args()
    opts.max_depth = 20  # FIXME get this from the data

    chats, trees = generate_actions(50000)
    word_dict, key_dict, intent_dict = build_dictionaries(chats, trees)
    opts.n_words = len(word_dict["i2w"]) + 1
    opts.n_keys = len(key_dict["i2w"]) + 1
    opts.null_word = len(word_dict["i2w"])
    opts.null_key = len(key_dict["i2w"])
    data = ActionDataset(opts, word_dict, key_dict, intent_dict)

    DL = torch.utils.data.DataLoader(
        data,
        batch_size=opts.batchsize,
        pin_memory=True,
        drop_last=False,
        collate_fn=batch_collater,
        num_workers=opts.ndonkeys,
    )
