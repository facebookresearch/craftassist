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


def viewer_pos_look_diection_to_glue_cubes_target_center(
    viewer_pos, viewer_look, direction, cube_size
):
    if direction[0] + direction[1] + direction[2] != 1:
        raise Exception("We need a dimension")
    if direction[3] + direction[4] != 1:
        raise Exception("We need a direction")

    vl = [i.item() for i in viewer_look]
    if direction[2] == 1:
        if direction[3] == 1:
            vl[2] += cube_size
            return torch.tensor(vl, dtype=torch.float)
        elif direction[4] == 1:
            vl[2] -= cube_size
            return torch.tensor(vl, dtype=torch.float)

    possible_targets = [
        [vl[0] + cube_size, vl[1], vl[2]],
        [vl[0] - cube_size, vl[1], vl[2]],
        [vl[0], vl[1] + cube_size, vl[2]],
        [vl[0], vl[1] - cube_size, vl[2]],
    ]
    possible_targets = [torch.tensor(l, dtype=torch.float) for l in possible_targets]

    # The vector we're going to rotate
    vl_to_ts = [get_vector(viewer_look[:2], t[:2]) for t in possible_targets]

    rotation_matrix = get_rotation_matrix(
        viewer_pos.unsqueeze(0), viewer_look.unsqueeze(0)
    ).squeeze(0)
    vl_to_t_rotated = [rotate_x_y(vl_to_t.double(), rotation_matrix) for vl_to_t in vl_to_ts]

    epsilon = 0.0001
    vals = []
    dim = 0 if direction[0] == 1 else 1

    vals = [v[dim] for v in vl_to_t_rotated]
    if direction[3] == 1:
        max_ind, dist = get_firstmax(vals, epsilon)
    else:
        max_ind, dist = get_firstmax(vals, epsilon, minlist=True)
    return possible_targets[max_ind]


def get_glue_cubes_cont_size(c_sl, s_sl, fixed_size=None, center=False):
    cube_size = fixed_size
    if not cube_size:
        cube_size = np.random.randint(1, s_sl + 1)  # +1 for inclusive

    if center:
        blc_cont = [c_sl // 2 - cube_size // 2 for i in range(3)]
    else:
        possible_range = [[cube_size, c_sl - 2 * cube_size] for i in range(3)]
        blc_cont = [np.random.randint(*possible_range[i]) for i in range(3)]
    return cube_size, blc_cont


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


# TODO: this is untested
def glue_cubes(c_sl, s_sl, dim=0, direction=1):
    cube_size, blc_cont = get_glue_cubes_cont_size(c_sl, s_sl)
    blc_seg = [i for i in blc_cont]
    blc_seg[dim] = blc_seg[dim] + (1 if direction > 1 else -1) * cube_size
    context_sparse, seg_sparse = get_sparse_cube_context_seg(cube_size, blc_cont, blc_seg)
    return context_sparse, seg_sparse


def dim_dir_to_direction_vector(dim, dr):
    output = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float)
    output[dim] = 1
    output[3 if dr > 0 else 4] = 1
    return output


def cube_center_to_blc(center, radius):
    return [i - radius for i in center]


def cube_blc_to_center(blc, radius):
    return [i + radius for i in blc]


def directional_glue_cubes(c_sl, s_sl, fixed_cube_size=None, fixed_center=False):
    cube_dim = random.choice([0, 1, 2])
    cube_direction = random.choice([-1, 1])
    direction_vec = dim_dir_to_direction_vector(cube_dim, cube_direction)

    cube_size, blc_cont = get_glue_cubes_cont_size(
        c_sl, s_sl, fixed_size=fixed_cube_size, center=fixed_center
    )
    cube_radius = cube_size // 2
    center_cont = cube_blc_to_center(blc_cont, cube_radius)

    viewer_pos = torch.tensor(random_int_triple(0, c_sl - 1), dtype=torch.float)
    viewer_look = torch.tensor(center_cont, dtype=torch.float)
    center_seg = viewer_pos_look_diection_to_glue_cubes_target_center(
        viewer_pos, viewer_look, direction_vec, cube_size
    )
    blc_seg = cube_center_to_blc(center_seg.int().tolist(), cube_radius)
    target_coord = torch.tensor(blc_seg, dtype=torch.float)
    viewer_info = [viewer_pos, viewer_look, direction_vec]

    context_sparse, seg_sparse = get_sparse_cube_context_seg(cube_size, blc_cont, blc_seg)
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
        fixed_cube_size=None,
        fixed_center=False,
    ):
        self.context_side_length = context_side_length
        self.seg_side_length = seg_side_length
        self.num_examples = nexamples
        self.useid = useid
        self.examples = []
        self.for_vis = for_vis
        self.use_direction = use_direction
        self.fixed_cube_size = fixed_cube_size
        self.fixed_center = fixed_center

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
                self.context_side_length,
                self.seg_side_length,
                self.fixed_cube_size,
                self.fixed_center,
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
