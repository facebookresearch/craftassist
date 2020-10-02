"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import numpy as np
import torch
import os
import sys
import random

CRAFTASSIST_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../")
TEST_DIR = os.path.join(CRAFTASSIST_DIR, "test/")
sys.path.append(CRAFTASSIST_DIR)
sys.path.append(TEST_DIR)

from world import World, Opt, flat_ground_generator

"""
Generic Spatial Utils
"""


def get_bounds(sparse_voxel):
    """
    Voxel should either be a schematic, a list of ((x, y, z), (block_id, ?)) objects
    or a list of coordinates.
    Returns a list of bounds.
    """
    if len(sparse_voxel) == 0:
        return [0, 0, 0, 0, 0, 0]

    # A schematic
    if len(sparse_voxel[0]) == 2 and len(sparse_voxel[0][0]) == 3 and len(sparse_voxel[0][1]) == 2:
        x, y, z = list(zip(*list(zip(*sparse_voxel))[0]))
    # A list or coordinates
    elif len(sparse_voxel[0]) == 3:
        x, y, z = list(zip(*sparse_voxel))
    else:
        raise Exception("Unknown schematic format")
    return min(x), max(x), min(y), max(y), min(z), max(z)


def get_side_lengths(bounds):
    """
    Bounds should be a list of [min_x, max_x, min_y, max_y, min_z, max_z].
    Returns a list of the side lengths.
    """
    return [x + 1 for x in (bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])]


def get_bounds_and_sizes(sparse_voxel):
    bounds = get_bounds(sparse_voxel)
    side_lengths = get_side_lengths(bounds)
    return bounds, side_lengths


def coord_to_index(coord, sl):
    """
    Takes a 3D coordinate in a cube and the cube side length.
    Returns index in flattened 3D array.
    """
    return coord[0] * sl * sl + coord[1] * sl + coord[2]


def index_to_coord(index, sl):
    """
    Takes an index into a flattened 3D array and its side length.
    Returns the coordinate in the cube.
    """
    coord = []
    two_d_slice_size = sl * sl
    coord.append(index // two_d_slice_size)
    remaining = index % two_d_slice_size
    coord.append(remaining // sl)
    coord.append(remaining % sl)
    return coord


def shift_sparse_voxel_to_origin(sparse_voxel):
    """
    Takes a segment, described as a list of tuples of the form:
        ((x, y, z), (block_id, ?))
    Returns the segment in the same form, shifted to the origin, and the shift vec
    """
    bounds = get_bounds(sparse_voxel)
    shift_zero_vec = [-bounds[0], -bounds[2], -bounds[4]]
    new_voxel = []
    for s in sparse_voxel:
        new_voxel.append((tuple([sum(x) for x in zip(s[0], shift_zero_vec)]), s[1]))
    return new_voxel, shift_zero_vec


# outputs a dense voxel rep (np array) from a sparse one.
# size should be a tuple of (H, W, D) for the desired voxel representation
# useid=True puts the block id into the voxel representation,
#    otherwise put a 1
def densify(blocks, size, center=(0, 0, 0), useid=False):
    V = np.zeros((size[0], size[1], size[2]), dtype="int32")

    offsets = (size[0] // 2 - center[0], size[1] // 2 - center[1], size[2] // 2 - center[2])
    for b in blocks:
        x = b[0][0] + offsets[0]
        y = b[0][1] + offsets[1]
        z = b[0][2] + offsets[2]
        if x >= 0 and y >= 0 and z >= 0 and x < size[0] and y < size[1] and z < size[2]:
            if type(b[1]) is int:
                V[x, y, z] = b[1]
            else:
                V[x, y, z] = b[1][0]
    if not useid:
        V[V > 0] = 1
    return V, offsets


def get_dense_array_from_sl(sparse_shape, sl, useid):
    center = [sl // 2, sl // 2, sl // 2]
    shape_dense, _ = np.asarray(densify(sparse_shape, [sl, sl, sl], center=center, useid=useid))
    return shape_dense


"""
Geoscorer Specific Spatial Utils
"""


def combine_seg_context(seg, context, seg_shift, seg_mult=1):
    c_sl = context.size()[0]
    s_sl = seg.size()[0]
    completed_context = context.clone()
    # Calculate the region to copy over, sometimes the segment
    #   falls outside the range of the context bounding box
    cs = [slice(s, min(s + s_sl, c_sl)) for s in seg_shift]
    ss = [slice(0, s_sl - max(0, s + s_sl - c_sl)) for s in seg_shift]
    completed_context[cs] = seg_mult * seg[ss] + context[cs]
    return completed_context


def convert_sparse_context_seg_to_target_coord_shifted_seg(context_sparse, seg_sparse, c_sl, s_sl):
    shifted_seg_sparse, shift_vec = shift_sparse_voxel_to_origin(seg_sparse)
    target_coord = [-x for x in shift_vec]
    return target_coord, shift_vec, shifted_seg_sparse


def convert_sparse_context_seg_target_to_example(
    context_sparse, shifted_seg_sparse, target_coord, c_sl, s_sl, useid, schem_sparse=None
):
    context_dense = get_dense_array_from_sl(context_sparse, c_sl, useid)
    seg_dense = get_dense_array_from_sl(shifted_seg_sparse, s_sl, useid)
    target_index = coord_to_index(target_coord, c_sl)
    example = {
        "context": torch.from_numpy(context_dense),
        "seg": torch.from_numpy(seg_dense),
        "target": torch.tensor([target_index]),
    }
    if schem_sparse:
        schem_dense = get_dense_array_from_sl(schem_sparse, c_sl, useid)
        example["schematic"] = torch.from_numpy(schem_dense)
    return example


def convert_sparse_context_seg_to_example(
    context_sparse, seg_sparse, c_sl, s_sl, useid, schem_sparse=None
):
    context_dense = get_dense_array_from_sl(context_sparse, c_sl, useid)
    shifted_seg_sparse, shift_vec = shift_sparse_voxel_to_origin(seg_sparse)
    seg_dense = get_dense_array_from_sl(shifted_seg_sparse, s_sl, useid)
    target_coord = [-x for x in shift_vec]
    target_index = coord_to_index(target_coord, c_sl)
    example = {
        "context": torch.from_numpy(context_dense),
        "seg": torch.from_numpy(seg_dense),
        "target": torch.tensor([target_index]),
    }
    if schem_sparse:
        schem_dense = get_dense_array_from_sl(schem_sparse, c_sl, useid)
        example["schematic"] = torch.from_numpy(schem_dense)
    return example


def add_ground_to_context(context_sparse, target_coord, flat=True, random_height=True):
    min_z = min([c[0][2] for c in context_sparse] + [target_coord[2].item()])
    max_ground_depth = min_z
    if max_ground_depth == 0:
        return
    if random_height:
        ground_depth = random.randint(1, max_ground_depth)
    else:
        ground_depth = max_ground_depth

    pos_z = 63

    shift = (-16, pos_z - 1 - ground_depth, -16)
    spec = {
        "players": [],
        "item_stacks": [],
        "mobs": [],
        "agent": {"pos": (0, pos_z, 0)},
        "coord_shift": shift,
    }
    world_opts = Opt()
    world_opts.sl = 32

    if flat or max_ground_depth == 1:
        spec["ground_generator"] = flat_ground_generator
        spec["ground_args"] = {"ground_depth": ground_depth}
    else:
        world_opts.avg_ground_height = max_ground_depth // 2
        world_opts.hill_scale = max_ground_depth // 2
    world = World(world_opts, spec)

    ground_blocks = []
    for l, d in world.blocks_to_dict().items():
        shifted_l = tuple([l[i] - shift[i] for i in range(3)])
        xzy_to_xyz = [shifted_l[0], shifted_l[2], shifted_l[1]]
        ground_blocks.append((xzy_to_xyz, d))
    context_sparse += ground_blocks
