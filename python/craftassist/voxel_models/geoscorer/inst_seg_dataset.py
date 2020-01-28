"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import os
import sys
import random
import pickle
import torch
from torch.utils import data as tds

GEOSCORER_DIR = os.path.dirname(os.path.realpath(__file__))
CRAFTASSIST_DIR = os.path.join(GEOSCORER_DIR, "../../")
sys.path.append(CRAFTASSIST_DIR)

from geoscorer_util import *


def load_segments(dpath, min_seg_size, save_path=None):
    inst_data = pickle.load(open(dpath, "rb"))
    all_segs = []
    schematics = []
    N = len(inst_data)
    num_segs_total = 0
    num_segs_kept = 0
    for j in range(N):
        h = inst_data[j]
        S, segs, _, _ = h
        segs = segs.astype("int32")
        blocks = list(zip(*S.nonzero()))
        schematics.append([tuple((b, tuple((S[b], 0)))) for b in blocks])
        instances = [
            [True if segs[blocks[p]] == q else False for p in range(len(blocks))]
            for q in range(1, segs.max() + 1)
        ]
        for i in instances:
            num_segs_total += 1
            if np.sum(np.asarray(i).astype("int32")) < min_seg_size:
                continue
            num_segs_kept += 1
            all_segs.append({"inst": i, "schematic": j})
    pretty_log("Num Segs {} Num Kept {}".format(num_segs_total, num_segs_kept))
    if save_path:
        save_specific_segments(save_path, all_segs, schematics)
    return all_segs, schematics


def load_specific_segments(dpath):
    specific_seg_data = pickle.load(open(dpath, "rb"))
    return specific_seg_data


def save_specific_segments(dpath, all_segs, schematic):
    to_save = [all_segs, schematic]
    with open(dpath, "wb") as f:
        pickle.dump(to_save, f)
    pretty_log("Saved segments and schematics to {}".format(dpath))


# Get a random 8x8x8 portion of the segment
def get_seg_bounded_sparse(seg_sparse, sl):
    bounds = get_bounds(seg_sparse)
    sizes = get_side_lengths(bounds)

    seg_bounded_sparse = []
    num_tries = 0
    while len(seg_bounded_sparse) == 0:
        new_starts = [bounds[0], bounds[2], bounds[4]]
        for i in range(3):
            if sizes[i] >= sl:
                diff = sizes[i] - sl
                new_starts[i] += random.randint(0, diff + 1)  # [0, diff] inclusive
        for s in seg_sparse:
            use = True
            for i in range(3):
                if s[0][i] < new_starts[i] or s[0][i] >= new_starts[i] + sl:
                    use = False
            if use:
                seg_bounded_sparse.append(s)
        num_tries += 1
        if num_tries > 20:
            print("num tries: {}".format(num_tries))
    return seg_bounded_sparse


def load_and_parse_segments(data_dir, min_seg_size, data_preparsed, save_preparsed):
    dpath = os.path.join(data_dir, "training_data.pkl")
    if data_preparsed:
        spath = os.path.join(data_dir, "training_data_min{}.pkl".format(min_seg_size))
        print("The path we should load from: {}".format(spath))
        all_segs, schematics = load_specific_segments(spath)
    else:
        all_segs, schematics = load_segments(dpath, min_seg_size)
        if save_preparsed:
            spath = os.path.join(data_dir, "training_data_min{}.pkl".format(min_seg_size))
            save_specific_segments(spath, all_segs, schematics)
    return all_segs, schematics


class SegmentContextCombinedInstanceData(tds.Dataset):
    def __init__(
        self,
        data_dir="/checkpoint/drotherm/minecraft_dataset/vision_training/training3/",
        nexamples=10000,
        sidelength=32,
        useid=False,
        nneg=5,
        shift_max=10,
        min_seg_size=6,
        ntype_probs=[1.0, 0.0, 0.0],
    ):
        self.sidelength = sidelength
        self.useid = useid
        self.nneg = nneg
        self.shift_max = shift_max
        self.nexamples = nexamples
        self.negative_sampler = NegativeSampler(
            self, shift_max=self.shift_max, ntype_probs=ntype_probs
        )
        self.examples = []
        dpath = os.path.join(data_dir, "training_data.pkl")
        self.all_segs, self.schematics = load_segments(dpath, min_seg_size)

    def toggle_useid(self):
        self.useid = not self.useid

    def _get_example(self):
        sl = self.sidelength
        inst = random.choice(self.all_segs)
        seg, s = inst["inst"], self.schematics[inst["schematic"]]
        c = center_of_mass(s, seg=seg)
        out = torch.zeros(self.nneg + 1, sl, sl, sl)
        p, _ = densify(s, [sl, sl, sl], center=c, useid=self.useid)
        out[0] = torch.tensor(p)
        for i in range(self.nneg):
            n = self.negative_sampler.build_negative(s, seg)
            out[i + 1] = torch.tensor(densify(n, [sl, sl, sl], center=c, useid=self.useid)[0])
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


# Returns three tensors: 32x32x32 context, 8x8x8 segment, 1 target
class SegmentContextSeparateInstanceData(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir="/checkpoint/drotherm/minecraft_dataset/vision_training/training3/",
        nexamples=10000,
        context_side_length=32,
        seg_side_length=8,
        useid=False,
        min_seg_size=6,
        data_preparsed=False,
        save_preparsed=False,
        for_vis=False,
    ):
        self.context_side_length = context_side_length
        self.seg_side_length = seg_side_length
        self.num_examples = nexamples
        self.useid = useid
        self.examples = []
        self.for_vis = for_vis
        print("Data preparsed {} Save Preparsed {}".format(data_preparsed, save_preparsed))
        self.all_segs, self.schematics = load_and_parse_segments(
            data_dir, min_seg_size, data_preparsed, save_preparsed
        )

    def toggle_useid(self):
        self.useid = not self.useid

    def _get_example(self):
        inst = random.choice(self.all_segs)
        seg, context_sparse = inst["inst"], self.schematics[inst["schematic"]]
        seg_sparse = sparsify_segment(seg, context_sparse)
        seg_bounded_sparse = get_seg_bounded_sparse(seg_sparse, self.seg_side_length)
        return convert_sparse_context_seg_to_example(
            context_sparse,
            seg_bounded_sparse,
            self.context_side_length,
            self.seg_side_length,
            self.useid,
            self.for_vis,
        )

    def __getitem__(self, index):
        return self._get_example()

    def __len__(self):
        return abs(self.num_examples)


# Returns a 32x32x32 context, 8x8x8 segment, 6 viewer, 1 direction, 1 target
class SegmentContextDirectionalInstanceData(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir="/checkpoint/drotherm/minecraft_dataset/vision_training/training3/",
        nexamples=10000,
        context_side_length=32,
        seg_side_length=8,
        useid=False,
        min_seg_size=6,
        data_preparsed=False,
        save_preparsed=False,
        for_vis=False,
    ):
        self.context_side_length = context_side_length
        self.seg_side_length = seg_side_length
        self.num_examples = nexamples
        self.useid = useid
        self.examples = []
        self.for_vis = for_vis
        print("Data preparsed {} Save Preparsed {}".format(data_preparsed, save_preparsed))
        self.all_segs, self.schematics = load_and_parse_segments(
            data_dir, min_seg_size, data_preparsed, save_preparsed
        )

    def _get_example(self):
        inst = random.choice(self.all_segs)
        seg, context_sparse = inst["inst"], self.schematics[inst["schematic"]]
        seg_sparse = sparsify_segment(seg, context_sparse)
        seg_bounded_sparse = get_seg_bounded_sparse(seg_sparse, self.seg_side_length)
        context, seg, target = convert_sparse_context_seg_to_example(
            context_sparse,
            seg_bounded_sparse,
            self.context_side_length,
            self.seg_side_length,
            self.useid,
            self.for_vis,
        )
        viewer_pos, viewer_look = get_random_viewer_info(self.context_side_length)
        target_coord = torch.tensor(index_to_coord(target, self.context_side_length))
        dir_vec = get_sampled_direction_vec(viewer_pos, viewer_look, target_coord)
        return [context, seg, target, viewer_pos, viewer_look, dir_vec]

    def __getitem__(self, index):
        return self._get_example()

    def __len__(self):
        return abs(self.num_examples)


if __name__ == "__main__":
    import os
    import argparse
    import visdom

    VOXEL_MODELS_DIR = os.path.join(GEOSCORER_DIR, "../../")
    sys.path.append(VOXEL_MODELS_DIR)
    import plot_voxels as pv

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, default="combined", help="(combined|separate)")
    parser.add_argument("--min_seg_size", type=int, default=6, help="min seg size")
    parser.add_argument(
        "--save_dataset", type=bool, default=False, help="should we save this min seg size dataset"
    )
    parser.add_argument(
        "--load_saved_dataset",
        type=bool,
        default=False,
        help="should we load a presaved dataset for this mss",
    )
    parser.add_argument("--useid", type=bool, default=False, help="should we use the block id")
    opts = parser.parse_args()

    vis = visdom.Visdom(server="http://localhost")
    sp = pv.SchematicPlotter(vis)

    if opts.data_type == "separate":
        dataset = SegmentContextSeparateInstanceData(
            nexamples=3,
            min_seg_size=opts.min_seg_size,
            data_preparsed=opts.load_saved_dataset,
            save_preparsed=opts.save_dataset,
            for_vis=True,
            useid=opts.useid,
        )
        for n in range(len(dataset)):
            shape, seg, target = dataset[n]
            sp.drawPlotly(shape)
            sp.drawPlotly(seg)
            target_coord = index_to_coord(target.item(), 32)
            completed_shape = combine_seg_context(seg, shape, target_coord, seg_mult=3)
            sp.drawPlotly(completed_shape)
    else:
        num_examples = 2
        num_neg = 2
        dataset = SegmentContextCombinedInstanceData(
            nexamples=num_examples, shift_max=10, min_seg_size=opts.min_seg_size, nneg=num_neg
        )
        for n in range(num_examples):
            curr_data = dataset[n]
            sp.drawPlotly(curr_data[0])
            for i in range(num_neg):
                sp.drawPlotly(curr_data[i + 1])
