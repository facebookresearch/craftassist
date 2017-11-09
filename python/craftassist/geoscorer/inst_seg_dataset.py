import os
import sys
import random
import pickle
import torch
from torch.utils import data as tds

from util import *

GEOSCORER_DIR = os.path.dirname(os.path.realpath(__file__))
CRAFTASSIST_DIR = os.path.join(GEOSCORER_DIR, "../")
sys.path.append(CRAFTASSIST_DIR)


class SegmentCenterInstanceData(tds.Dataset):
    def __init__(
        self,
        data_dir="/private/home/aszlam/haonan/",
        nexamples=10000,
        sidelength=32,
        useid=False,
        nneg=5,
        shift_max=10,
    ):
        self.sidelength = sidelength
        self.useid = useid
        self.nneg = nneg
        self.shift_max = shift_max
        self.examples = []
        dpath = os.path.join(data_dir, "training_data.pkl")
        self.inst_data = pickle.load(open(dpath, "rb"))
        self.nexamples = nexamples
        self.all_segs = []
        self.schematics = []
        N = len(self.inst_data)
        if self.nexamples < 0:
            N = min(N, -self.nexamples)
        self.negative_sampler = NegativeSampler(self, shift_max=self.shift_max)
        for j in range(N):
            h = self.inst_data[j]
            S, segs, _, _ = h
            segs = segs.astype("int32")
            blocks = list(zip(*S.nonzero()))
            self.schematics.append([tuple((b, tuple((S[b], 0)))) for b in blocks])
            instances = [
                [True if segs[blocks[p]] == q else False for p in range(len(blocks))]
                for q in range(1, segs.max() + 1)
            ]
            for i in instances:
                self.all_segs.append({"inst": i, "schematic": j})
        if self.nexamples < 0:
            for i in range(-self.nexamples):
                self.examples.append(self._get_example())

    def toggle_useid(self):
        self.useid = not self.useid

    def get_positive(self):
        inst = random.choice(self.all_segs)
        seg, S = inst["inst"], self.schematics[inst["schematic"]]
        return seg, S

    def _get_example(self):
        sl = self.sidelength
        inst = random.choice(self.all_segs)
        seg, S = inst["inst"], self.schematics[inst["schematic"]]
        c = center_of_mass(S, seg=seg)
        out = torch.zeros(self.nneg + 1, sl, sl, sl)
        P, _ = densify(S, [sl, sl, sl], center=c, useid=self.useid)
        out[0] = torch.Tensor(P)
        for i in range(self.nneg):
            N = self.negative_sampler.build_negative(S, seg)
            out[i + 1] = torch.Tensor(densify(N, [sl, sl, sl], center=c, useid=self.useid)[0])
        return out

    def __getitem__(self, index):
        if len(self.examples) > 0:
            #            sl = self.sidelength
            #            out = torch.zeros(self.nneg + 1, sl, sl, sl)
            #            out[1:] = 1
            #            return out
            return self.examples[index]
        else:
            return self._get_example()

    def __len__(self):
        return abs(self.nexamples)


if __name__ == "__main__":
    import os
    import argparse
    import visdom

    vis = visdom.Visdom(server="http://localhost")
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=32)
    opts = parser.parse_args()

    D = SegmentCenterInstanceData()
