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

import torch
import torch.utils.data
from geoscorer_util import *


def get_glue_cubes_coords_size(c_sl, s_sl, dim=0, direction=1):
    if direction not in [-1, 1]:
        raise Exception("Direction must be -1 or 1 for glue cubes")
    if dim < 0 or dim > 3:
        raise Exception("Dimension must be 0, 1, or 2 for glue cubes")
    cube_size = np.random.randint(1, s_sl + 1)  # +1 for inclusive

    # leave space for context cube
    possible_range = [[0, c_sl - cube_size] for i in range(3)]
    # leave space for segment cube
    if direction == -1:
        possible_range[dim][0] += cube_size
    elif direction == 1:
        possible_range[dim][1] -= cube_size

    blc_cont = [np.random.randint(*possible_range[i]) for i in range(3)]
    blc_seg = [i for i in blc_cont]
    blc_seg[dim] = blc_seg[dim] + direction * cube_size
    return cube_size, blc_cont, blc_seg


def get_sparse_cube_context_seg(cube_size, blc_cont, blc_seg, block_type=None):
    context_sparse = []
    seg_sparse = []
    cube_type = block_type if block_type else np.random.randint(1, 256)
    for i in range(cube_size):
        for j in range(cube_size):
            for k in range(cube_size):
                offset = (i, j, k)
                context_sparse.append(
                    (tuple([sum(x) for x in zip(offset, blc_cont)]), (cube_type, 0))
                )
                seg_sparse.append((tuple([sum(x) for x in zip(offset, blc_seg)]), (cube_type, 0)))
    context_sparse += seg_sparse
    return context_sparse, seg_sparse


def glue_cubes(c_sl, s_sl, dim=0, direction=1):
    cube_size, blc_cont, blc_seg = get_glue_cubes_coords_size(c_sl, s_sl, dim, direction)
    context_sparse, seg_sparse = get_sparse_cube_context_seg(cube_size, blc_cont, blc_seg)
    return context_sparse, seg_sparse


def directional_glue_cubes(c_sl, s_sl):
    cube_dim = random.choice([0, 1, 2])
    cube_direction = random.choice([-1, 1])
    cube_size, blc_cont, blc_seg = get_glue_cubes_coords_size(c_sl, s_sl, cube_dim, cube_direction)
    context_sparse, seg_sparse = get_sparse_cube_context_seg(cube_size, blc_cont, blc_seg)

    viewer_pos = torch.tensor(random_int_triple(0, c_sl - 1), dtype=torch.float)
    # Have viewer look at the center of the context cube
    cube_radius = cube_size * 0.5
    viewer_look = torch.tensor([i + cube_radius for i in blc_cont], dtype=torch.float)
    viewer_dir = get_dir_vec(viewer_pos, viewer_look)
    target_coord = torch.tensor(blc_seg, dtype=torch.float)
    dir_vector = get_sampled_dir(viewer_pos, viewer_look, target_coord)
    viewer_info = [viewer_pos, viewer_look, viewer_dir, dir_vector]

    return context_sparse, seg_sparse, target_coord, viewer_info


# Returns three tensors: 32x32x32 context, 8x8x8 segment, 1 target
class SegmentContextSeparateAutogenData(torch.utils.data.Dataset):
    def __init__(
        self,
        nexamples=100000,
        context_side_length=32,
        seg_side_length=8,
        useid=False,
        for_vis=False,
        type_name="random",
        use_direction=False,
    ):
        self.context_side_length = context_side_length
        self.seg_side_length = seg_side_length
        self.num_examples = nexamples
        self.useid = useid
        self.examples = []
        self.for_vis = for_vis
        self.use_direction = use_direction

        data_gens = {"glue_cubes": glue_cubes}
        self.data_gen = data_gens.get(type_name, random.choice(list(data_gens.values())))

    def _get_example(self):
        if not self.use_direction:
            context_sparse, seg_sparse = self.data_gen(
                self.context_side_length, self.seg_side_length
            )
            return convert_sparse_context_seg_to_example(
                context_sparse,
                seg_sparse,
                self.context_side_length,
                self.seg_side_length,
                self.useid,
                self.for_vis,
            )
        else:
            context_sparse, seg_sparse, target_coord, viewer_info = directional_glue_cubes(
                self.context_side_length, self.seg_side_length
            )
            context, seg, _ = convert_sparse_context_seg_to_example(
                context_sparse,
                seg_sparse,
                self.context_side_length,
                self.seg_side_length,
                self.useid,
                self.for_vis,
            )
            target = coord_to_index(target_coord, self.context_side_length)
            return [context, seg, target.unsqueeze(0)] + viewer_info

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

    vis = visdom.Visdom(server="http://localhost")
    sp = pv.SchematicPlotter(vis)

    dataset = SegmentContextSeparateAutogenData(nexamples=3, for_vis=True)
    for n in range(len(dataset)):
        shape, seg, target = dataset[n]
        sp.drawPlotly(shape)
        sp.drawPlotly(seg)
        target_coord = index_to_coord(target.item(), 32)
        completed_shape = combine_seg_context(seg, shape, target_coord, seg_mult=3)
        sp.drawPlotly(completed_shape)
