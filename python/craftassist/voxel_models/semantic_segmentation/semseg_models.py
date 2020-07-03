"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import numpy as np
import torch
import pickle
import torch.nn as nn
from data_loaders import make_example_from_raw


class SemSegNet(nn.Module):
    def __init__(self, opts, classes=None):
        super(SemSegNet, self).__init__()

        if opts.load:
            if opts.load_model != "":
                self.load(opts.load_model)
            else:
                raise ("loading from file specified but no load_filepath specified")

            if opts.vocab_path != "":
                self.load_vocab(opts.vocab_path)
            else:
                raise ("loading from file specified but no vocab_path specified")
        else:
            self.opts = opts
            self._build()
            self.classes = classes

    def _build(self):
        opts = self.opts
        try:
            embedding_dim = opts.embedding_dim
        except:
            embedding_dim = 8
        try:
            num_words = opts.num_words
        except:
            num_words = 3
        try:
            num_layers = opts.num_layers
        except:
            num_layers = 4  # 32x32x32 input
        try:
            hidden_dim = opts.hidden_dim
        except:
            hidden_dim = 64

        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_words, embedding_dim)
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.layers.append(
            nn.Sequential(
                nn.Conv3d(embedding_dim, hidden_dim, kernel_size=5, padding=2),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU(inplace=True),
            )
        )
        for i in range(num_layers - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Conv3d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                    nn.BatchNorm3d(hidden_dim),
                    nn.ReLU(inplace=True),
                )
            )
        self.out = nn.Conv3d(hidden_dim, opts.num_classes, kernel_size=1)
        self.lsm = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # FIXME when pytorch is ready for this, embedding
        # backwards is soooooo slow
        # z = self.embedding(x)
        szs = list(x.size())
        x = x.view(-1)
        z = self.embedding.weight.index_select(0, x)
        szs.append(self.embedding_dim)
        z = z.view(torch.Size(szs))
        z = z.permute(0, 4, 1, 2, 3).contiguous()
        for i in range(self.num_layers):
            z = self.layers[i](z)
        return self.lsm(self.out(z))

    def save(self, filepath):
        self.cpu()
        sds = {}
        sds["opts"] = self.opts
        sds["classes"] = self.classes
        sds["state_dict"] = self.state_dict()
        torch.save(sds, filepath)
        if self.opts.cuda:
            self.cuda()

    def load_vocab(self, vocab_path):
        with open(vocab_path, "rb") as file:
            self.vocab = pickle.load(file)
        print("Loaded vocab")

    def load(self, filepath):
        sds = torch.load(filepath)
        self.opts = sds["opts"]
        print("loading from file, using opts")
        print(self.opts)
        self._build()
        self.load_state_dict(sds["state_dict"])
        self.zero_grad()
        self.classes = sds["classes"]


class Opt:
    pass


class SemSegWrapper:
    def __init__(self, model, vocab_path, threshold=-1.0, blocks_only=True, cuda=False):
        if type(model) is str:
            opts = Opt()
            opts.load = True
            opts.load_model = model
            opts.vocab_path = vocab_path
            model = SemSegNet(opts)
        self.model = model
        self.cuda = cuda
        if self.cuda:
            model.cuda()
        else:
            model.cpu()
        self.classes = model.classes
        # threshold for relevance; unused rn
        self.threshold = threshold
        # if true only label non-air blocks
        self.blocks_only = blocks_only
        # this is used by the semseg_process
        i2n = self.classes["idx2name"]
        self.tags = [(c, self.classes["name2count"][c]) for c in i2n]
        assert self.classes["name2idx"]["none"] == 0

    @torch.no_grad()
    def segment_object(self, blocks):
        self.model.eval()

        if self.model.vocab:

            vocab = self.model.vocab
            vocab_blocks = np.zeros(blocks.shape[:-1])
            for x in range(blocks.shape[0]):
                for y in range(blocks.shape[1]):
                    for z in range(blocks.shape[2]):
                        block_id = blocks[x,y,z,0]
                        meta_id = blocks[x,y,z,1]
                        id_tuple = (block_id, meta_id)
                        # First see if that specific block-meta pair is in the vocab.
                        if id_tuple in vocab:
                            id_ = vocab[id_tuple]
                        # Else, check if the same general material (block-id) exists.
                        elif (block_id, 0) in vocab:
                            id_ = vocab[(block_id, 0)]
                        # If not, the network has no clue what it is, ignore it (treat as air).
                        else:
                            id_ = vocab[(0,0)]
                        
                        vocab_blocks[x,y,z] = id_
        else:
            vocab_blocks = blocks[:, :, :, 0]

        blocks = torch.from_numpy(vocab_blocks)
        blocks, _, o = make_example_from_raw(blocks)
        blocks = blocks.unsqueeze(0)
        if self.cuda:
            blocks = blocks.cuda()
        y = self.model(blocks)
        _, mids = y.squeeze().max(0)
        locs = mids.nonzero()
        locs = locs.tolist()
        if self.blocks_only:
            return {
                tuple(np.subtract(l, o)): mids[l[0], l[1], l[2]].item()
                for l in locs
                if blocks[0, l[0], l[1], l[2]] > 0
            }
        else:
            return {tuple(ll for ll in l): mids[l[0], l[1], l[2]].item() for l in locs}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="size of hidden dim in fc layer"
    )
    parser.add_argument("--embedding_dim", type=int, default=16, help="size of blockid embedding")
    parser.add_argument("--num_words", type=int, default=256, help="number of blocks")
    parser.add_argument("--num_classes", type=int, default=20, help="number of blocks")

    args = parser.parse_args()

    N = SemSegNet(args)
