"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import numpy as np
import random
import torch
import torch.utils.data
import spatial_utils as su
import directional_utils as du


def get_glue_cubes_direction_target_coord(viewer_pos, dir_vec, cube_size, origin_cont, c_sl):
    # Note: c_sizes and s_sizes are the same for this dataset
    c_sizes = [cube_size for _ in range(3)]
    target, _, _ = du.get_rotated_context_to_seg_origin(
        viewer_pos, dir_vec, origin_cont, c_sizes, c_sizes
    )
    return target


def get_glue_cubes_cont_size_loc(c_sl, s_sl, fixed_size=None, center=False):
    cube_size = fixed_size
    if not cube_size:
        cube_size = np.random.randint(1, s_sl + 1)  # +1 for inclusive

    if center:
        origin_cont = [c_sl // 2 - cube_size // 2 for i in range(3)]
    else:
        possible_range = [[cube_size, c_sl - 2 * cube_size] for i in range(3)]
        origin_cont = [np.random.randint(*possible_range[i]) for i in range(3)]
    return cube_size, origin_cont


def get_sparse_cube_context_seg(cube_size, origin_cont, origin_seg, block_type=None):
    context_sparse = []
    seg_sparse = []
    cube_type = block_type if block_type else np.random.randint(1, 256)
    for i in range(cube_size):
        for j in range(cube_size):
            for k in range(cube_size):
                offset = (i, j, k)
                context_sparse.append(
                    (tuple([sum(x) for x in zip(offset, origin_cont)]), (cube_type, 0))
                )
                seg_sparse.append(
                    (tuple([sum(x) for x in zip(offset, origin_seg)]), (cube_type, 0))
                )
    return context_sparse, seg_sparse


def glue_cubes(c_sl, s_sl, dim, direction):
    if dim < 0 or dim > 2 or direction not in [-1, 1]:
        raise Exception("Invalid dimension {} or direction {}".format(dim, direction))
    cube_size, origin_cont = get_glue_cubes_cont_size_loc(c_sl, s_sl)
    origin_seg = [i for i in origin_cont]
    origin_seg[dim] = origin_seg[dim] + direction * cube_size
    context_sparse, seg_sparse = get_sparse_cube_context_seg(cube_size, origin_cont, origin_seg)
    return context_sparse, seg_sparse


def directional_glue_cubes(c_sl, s_sl, fixed_cube_size=None, fixed_center=False):
    dir_vec = du.random_dir_vec_tensor()
    viewer_pos, viewer_look = du.get_random_viewer_info(c_sl)
    cube_size, origin_cont = get_glue_cubes_cont_size_loc(
        c_sl, s_sl, fixed_size=fixed_cube_size, center=fixed_center
    )
    target_coord = get_glue_cubes_direction_target_coord(
        viewer_pos, dir_vec, cube_size, origin_cont, c_sl
    )
    context_sparse, seg_sparse = get_sparse_cube_context_seg(
        cube_size, origin_cont, target_coord.tolist()
    )
    return {
        "context_sparse": context_sparse,
        "seg_sparse": seg_sparse,
        "target_coord": target_coord,
        "viewer_pos": viewer_pos,
        "dir_vec": dir_vec,
    }


# Returns three tensors: 32x32x32 context, 8x8x8 segment, 1 target
class SegmentContextGlueCubesData(torch.utils.data.Dataset):
    def __init__(
        self,
        nexamples=100000,
        context_side_length=32,
        seg_side_length=8,
        useid=False,
        type_name="random",
        use_direction=False,
        fixed_cube_size=None,
        fixed_center=False,
    ):
        self.c_sl = context_side_length
        self.s_sl = seg_side_length
        self.num_examples = nexamples
        self.useid = useid
        self.examples = []
        self.use_direction = use_direction
        self.fixed_cube_size = fixed_cube_size
        self.fixed_center = fixed_center

    def _get_example(self):
        if not self.use_direction:
            dim = random.choice([0, 1, 2])
            dr = random.choice([-1, 1])
            context_sparse, seg_sparse = glue_cubes(self.c_sl, self.s_sl, dim, dr)
            return su.convert_sparse_context_seg_to_example(
                context_sparse, seg_sparse, self.c_sl, self.s_sl, self.useid
            )
        else:
            dgc = directional_glue_cubes(
                self.c_sl, self.s_sl, self.fixed_cube_size, self.fixed_center
            )
            example = su.convert_sparse_context_seg_to_example(
                dgc["context_sparse"], dgc["seg_sparse"], self.c_sl, self.s_sl, self.useid
            )
            example["target"] = su.coord_to_index(dgc["target_coord"], self.c_sl)
            example["target"] = example["target"].unsqueeze(0)
            example["viewer_pos"] = dgc["viewer_pos"]
            example["dir_vec"] = dgc["dir_vec"]
            return example

    def __getitem__(self, index):
        return self._get_example()

    def __len__(self):
        return self.num_examples


if __name__ == "__main__":
    import argparse
    from visualization_utils import GeoscorerDatasetVisualizer

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_direction", action="store_true", help="use direction in example creation"
    )
    parser.add_argument(
        "--fixed_center", action="store_true", help="fix the center of the context cube"
    )
    parser.add_argument(
        "--fixed_cube_size", type=int, default=None, help="fix the size of the cubes"
    )
    opts = parser.parse_args()

    dataset = SegmentContextGlueCubesData(
        nexamples=3,
        use_direction=opts.use_direction,
        fixed_center=opts.fixed_center,
        fixed_cube_size=opts.fixed_cube_size,
    )
    vis = GeoscorerDatasetVisualizer(dataset)
    for n in range(len(dataset)):
        vis.visualize()
