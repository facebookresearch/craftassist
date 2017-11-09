import numpy as np
import os
import sys
import random

GEOSCORER_DIR = os.path.dirname(os.path.realpath(__file__))
CRAFTASSIST_DIR = os.path.join(GEOSCORER_DIR, "../")
sys.path.append(CRAFTASSIST_DIR)

import shapes
import shape_helpers as sh
import torch
import torchvision.utils as tvu
from util import *

# subshapes by everything in a l1 or l2 ball from a point.
# put pairs + triples of shapes in frame, sometimes one partially built


PERM = torch.randperm(256)
r = np.arange(0, 256) / 256
CMAP = np.stack((r, np.roll(r, 80), np.roll(r, 160)))


SMALL = 5
LARGE = 20


def get_shape():
    name = random.choice(SHAPENAMES)
    opts = SHAPE_HELPERS[name]()
    opts["labelme"] = False
    return SHAPEFNS[name](**opts), opts, name


def options_cube():
    return {"size": np.random.randint(SMALL, LARGE)}


def options_hollow_cube():
    opts = {}
    opts["size"] = np.random.randint(SMALL, LARGE)
    opts["thickness"] = np.random.randint(1, opts["size"] - 3)
    return opts


def options_rectanguloid():
    return {"size": np.random.randint(SMALL, LARGE, size=3)}


def options_hollow_rectanguloid():
    opts = {}
    opts["size"] = np.random.randint(SMALL, LARGE, size=3)
    ms = min(opts["size"])
    opts["thickness"] = np.random.randint(1, ms - 3)
    return opts


def options_sphere():
    return {"radius": np.random.randint(SMALL, LARGE)}


def options_spherical_shell():
    opts = {}
    opts["radius"] = np.random.randint(SMALL, LARGE)
    opts["thickness"] = np.random.randint(1, opts["radius"] - 3)
    return opts


def options_square_pyramid():
    opts = {}
    opts["radius"] = np.random.randint(SMALL, LARGE)
    opts["slope"] = np.random.rand() * 0.4 + 0.8
    fullheight = opts["radius"] * opts["slope"]
    opts["height"] = np.random.randint(0.5 * fullheight, fullheight)
    return opts


def options_square():
    return {"size": np.random.randint(SMALL, LARGE), "orient": sh.orientation3()}


def options_rectangle():
    return {"size": np.random.randint(SMALL, LARGE, size=2), "orient": sh.orientation3()}


def options_circle():
    return {"radius": np.random.randint(SMALL, LARGE), "orient": sh.orientation3()}


def options_disk():
    return {"radius": np.random.randint(SMALL, LARGE), "orient": sh.orientation3()}


def options_triangle():
    return {"size": np.random.randint(SMALL, LARGE), "orient": sh.orientation3()}


def options_dome():
    return {"radius": np.random.randint(SMALL, LARGE)}


def options_arch():
    return {"size": np.random.randint(SMALL, LARGE), "distance": 2 * np.random.randint(2, 5) + 1}


def options_ellipsoid():
    return {"size": np.random.randint(SMALL, LARGE, size=3)}


def options_tower():
    return {"height": np.random.randint(3, 30), "base": np.random.randint(-4, 6)}


def options_empty():
    return {}


def empty(labelme=False):
    num = np.random.randint(1, 64)
    S = []
    for i in range(num):
        pos = np.random.randint(0, 32, 3)
        bid = np.random.randint(0, 64)
        S.append((pos, bid))
    return S, []


# eventually put ground blocks, add 'floating', 'hill', etc.
# TODO hollow is separate tag
SHAPENAMES = sh.SHAPE_NAMES
SHAPENAMES.append("TOWER")
# SHAPENAMES.append("empty")
SHAPEFNS = sh.SHAPE_FNS
SHAPEFNS["TOWER"] = shapes.tower
SHAPEFNS["empty"] = empty

SHAPE_HELPERS = {
    "CUBE": options_cube,
    "HOLLOW_CUBE": options_hollow_cube,
    "RECTANGULOID": options_rectanguloid,
    "HOLLOW_RECTANGULOID": options_hollow_rectanguloid,
    "SPHERE": options_sphere,
    "SPHERICAL_SHELL": options_spherical_shell,
    "PYRAMID": options_square_pyramid,
    "SQUARE": options_square,
    "RECTANGLE": options_rectangle,
    "CIRCLE": options_circle,
    "DISK": options_disk,
    "TRIANGLE": options_triangle,
    "DOME": options_dome,
    "ARCH": options_arch,
    "ELLIPSOID": options_ellipsoid,
    "TOWER": options_tower,
    "empty": options_empty,
}

################################################################################
################################################################################


def draw_slices(c, L=None, vis=None):
    X = np.zeros((c.shape[0], 3, c.shape[1], c.shape[2]))
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            for k in range(c.shape[2]):
                for l in range(3):
                    X[j, l, i, k] = CMAP[l, PERM[int(c[i, j, k])]]
    img = tvu.make_grid(torch.from_numpy(X), nrow=8, normalize=True, scale_each=False)
    vis.image(img=img)
    if L is not None:
        Y = np.zeros((L[0].shape[0], 3, L[0].shape[1], L[0].shape[2]))
        idx = np.transpose(np.nonzero(L[0]))
        for i in idx:
            for l in range(3):
                Y[i[1], l, i[0], i[2]] += CMAP[l, PERM[int(i[3])]]
        img = tvu.make_grid(torch.from_numpy(Y), nrow=4, normalize=True, scale_each=False)
        vis.image(img=img)
        return X, Y
    else:
        return X


def get_rectanguloid_subsegment(S, c, max_chunk=10):
    bounds = shapes.get_bounds(S)
    dX = bounds[1] - bounds[0]
    dY = bounds[3] - bounds[2]
    dZ = bounds[5] - bounds[4]
    # don't allow the whole thing as subsegment
    dh = min(dX / 2, dY / 2, dZ / 2)
    max_chunk = round(min(max_chunk, dh))
    if max_chunk > 1:
        chunk_szs = random_int_triple(1, max_chunk)
    else:
        chunk_szs = [1, 1, 1]
    return [check_l1_dist(c, b[0], chunk_szs) for b in S]


class SegmentCenterShapeData(torch.utils.data.Dataset):
    def __init__(self, nexamples=10000, sidelength=32, useid=False, nneg=5, shift_max=10):
        self.sidelength = sidelength
        self.useid = useid
        self.nneg = nneg
        self.shift_max = shift_max
        self.examples = []
        if nexamples < 0:
            nexamples = -nexamples
            for i in range(nexamples):
                self.examples.append(self._get_example())
        self.nexamples = nexamples

    def _get_example(self):
        sl = self.sidelength
        S, o, name = get_shape()
        p = random.choice(S)[0]
        seg = get_rectanguloid_subsegment(S, p, max_chunk=10)
        c = center_of_mass(S, seg=seg)
        out = torch.zeros(self.nneg + 1, sl, sl, sl)
        P, _ = densify(S, [sl, sl, sl], center=c)
        out[0] = torch.Tensor(P)
        for i in range(self.nneg):
            N = build_negative(S, seg, shift_max=self.shift_max)
            out[i + 1] = torch.Tensor(densify(N, [sl, sl, sl], center=c)[0])
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
        return self.nexamples


if __name__ == "__main__":
    import os
    import argparse
    import visdom

    vis = visdom.Visdom(server="http://localhost")
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=32)
    opts = parser.parse_args()

    def build():
        S, o, name = get_shape()
        p = random.choice(S)[0]
        seg = get_rectanguloid_subsegment(S, p, max_chunk=10)
        c = center_of_mass(S, seg=seg)
        print(c)
        S = [(S[i][0], (2, 0)) if seg[i] else (S[i][0], (1, 0)) for i in range(len(S))]
        P, off = densify(S, [32, 32, 32], center=c, useid=True)
        nS = build_shift_negative(S, seg, 10)
        N, off = densify(nS, [32, 32, 32], center=c, useid=True)
        return P, N, S, nS

    def build_and_draw():
        P, N, S, nS = build()
        draw_slices(P, vis=vis)
        draw_slices(N, vis=vis)
        return P, N, S, nS

    build()


#    D = ShapeDataset()
