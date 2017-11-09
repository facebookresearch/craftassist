"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import numpy as np
import os
import sys
import random

GEOSCORER_DIR = os.path.dirname(os.path.realpath(__file__))
CRAFTASSIST_DIR = os.path.join(GEOSCORER_DIR, "../../")
sys.path.append(CRAFTASSIST_DIR)

import shapes
import shape_helpers as sh
import torch
import torch.utils.data
from geoscorer_util import *

# subshapes by everything in a l1 or l2 ball from a point.
# put pairs + triples of shapes in frame, sometimes one partially built


PERM = torch.randperm(256)
r = np.arange(0, 256) / 256
CMAP = np.stack((r, np.roll(r, 80), np.roll(r, 160)))


SMALL = 5
LARGE = 20


def get_shape(name="random", opts=None):
    if name != "random" and name not in SHAPENAMES:
        pretty_log("Shape name {} not in dict, choosing randomly".format(name))
        name = "random"
    if name == "random":
        name = random.choice(SHAPENAMES)
    if not opts:
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
def get_rectanguloid_subsegment(S, c, max_chunk=10):
    bounds = shapes.get_bounds(S)
    segment_sizes = get_side_lengths(bounds)
    max_dists = []
    for s in segment_sizes:
        max_side_len = min(s - 1, max_chunk)
        max_dist = int(max(max_side_len / 2, 1))
        max_dists.append(random.randint(1, max_dist))

    return [check_l1_dist(c, b[0], max_dists) for b in S]


def get_random_shape_pt(shape, side_length=None):
    sl = side_length
    p = random.choice(shape)[0]
    if not side_length:
        return p
    while p[0] >= sl or p[1] >= sl or p[2] >= sl:
        p = random.choice(shape)[0]
    return p


# Return values:
#   shape: array of ((3D coords), (block_id, ??))
#   seg: array of bool for whether pos is in seg
def get_shape_segment(max_chunk=10, side_length=None):
    shape, _, _ = get_shape()
    p = get_random_shape_pt(shape, side_length)

    seg = get_rectanguloid_subsegment(shape, p, max_chunk=max_chunk)
    s_tries = 0
    while all(item for item in seg):
        p = get_random_shape_pt(shape, side_length)
        seg = get_rectanguloid_subsegment(shape, p, max_chunk=max_chunk)
        s_tries += 1
        # Get new shape
        if s_tries > 3:
            shape, _, _ = get_shape()
            p = get_random_shape_pt(shape, side_length)
            seg = get_rectanguloid_subsegment(shape, p, max_chunk=max_chunk)
            s_tries = 0
    return shape, seg


def shift_vector_gen(side_length):
    shift_max = int(side_length / 2)
    for i in range(side_length):
        for j in range(side_length):
            for k in range(side_length):
                yield (i - shift_max, j - shift_max, k - shift_max)


class SegmentContextCombinedShapeData(torch.utils.data.Dataset):
    def __init__(
        self, nexamples=100000, sidelength=32, useid=False, nneg=1, shift_max=10, for_vis=False
    ):
        self.sidelength = sidelength
        self.useid = useid
        self.nneg = nneg
        self.shift_max = shift_max
        self.for_vis = for_vis
        self.examples = []
        if nexamples < 0:
            nexamples = -nexamples
            for i in range(nexamples):
                self.examples.append(self._get_example())
        self.nexamples = nexamples

    def _get_example(self):
        sl = self.sidelength
        shape, seg = get_shape_segment(max_chunk=10)
        center = center_of_mass(shape, seg=seg)
        out = torch.zeros(self.nneg + 1, sl, sl, sl)
        if self.for_vis:
            id_shape = shift_negative(shape, seg, {"shift_max": 0, "seg_id": 2})
            context_dense, _ = densify(id_shape, [sl, sl, sl], center=center, useid=self.useid)
        else:
            context_dense, _ = densify(shape, [sl, sl, sl], center=center, useid=self.useid)

        out[0] = torch.Tensor(context_dense)
        for i in range(self.nneg):
            shift_args = {"shift_max": self.shift_max}
            if self.for_vis:
                shift_args["seg_id"] = 2
            negative = shift_negative(shape, seg, shift_args)
            neg_dense = densify(negative, [sl, sl, sl], center=center, useid=self.useid)[0]
            out[i + 1] = torch.Tensor(neg_dense)
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


# Returns three tensors: 32x32x32 context, 8x8x8 segment, 1 target
class SegmentContextSeparateShapeData(torch.utils.data.Dataset):
    def __init__(
        self,
        nexamples=100000,
        context_side_length=32,
        seg_side_length=8,
        useid=False,
        for_vis=False,
    ):
        self.context_side_length = context_side_length
        self.seg_side_length = seg_side_length
        self.num_examples = nexamples
        self.useid = useid
        self.examples = []
        self.for_vis = for_vis

    def _get_example(self):
        context_sparse, seg = get_shape_segment(
            max_chunk=self.seg_side_length - 1, side_length=self.context_side_length
        )
        seg_sparse = sparsify_segment(seg, context_sparse)
        return convert_sparse_context_seg_to_example(
            context_sparse,
            seg_sparse,
            self.context_side_length,
            self.seg_side_length,
            self.useid,
            self.for_vis,
        )

    def __getitem__(self, index):
        return self._get_example()

    def __len__(self):
        return self.num_examples


if __name__ == "__main__":
    import os
    import visdom

    VOXEL_MODELS_DIR = os.path.join(GEOSCORER_DIR, "../../")
    sys.path.append(VOXEL_MODELS_DIR)
    import plot_voxels as pv
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, default="combined", help="(combined|separate)")
    opts = parser.parse_args()

    vis = visdom.Visdom(server="http://localhost")
    sp = pv.SchematicPlotter(vis)

    if opts.data_type == "separate":
        dataset = SegmentContextSeparateShapeData(nexamples=3, for_vis=True)
        for n in range(len(dataset)):
            shape, seg, target = dataset[n]
            sp.drawPlotly(shape)
            sp.drawPlotly(seg)
            target_coord = index_to_coord(target.item(), 32)
            completed_shape = combine_seg_context(seg, shape, target_coord, seg_mult=3)
            sp.drawPlotly(completed_shape)
    else:
        num_examples = 4
        num_neg = 3
        dataset = SegmentContextCombinedShapeData(
            nexamples=num_examples, for_vis=True, useid=True, shift_max=10, nneg=num_neg
        )
        for n in range(num_examples):
            curr_data = dataset[n]
            sp.drawPlotly(curr_data[0])
            for i in range(num_neg):
                sp.drawPlotly(curr_data[i + 1])
