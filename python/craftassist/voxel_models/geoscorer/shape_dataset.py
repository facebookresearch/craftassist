"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import numpy as np
import os
import sys
import random
import torch
import torch.utils.data

CRAFTASSIST_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../")
sys.path.append(CRAFTASSIST_DIR)

import shapes
import shape_helpers as sh
import spatial_utils as su
import directional_utils as du

# subshapes by everything in a l1 or l2 ball from a point.
# put pairs + triples of shapes in frame, sometimes one partially built


PERM = torch.randperm(256)
r = np.arange(0, 256) / 256
CMAP = np.stack((r, np.roll(r, 80), np.roll(r, 160)))


MIN_SIZE = 4


def get_shape(name="random", max_size=20, opts=None):
    if name != "random" and name not in SHAPENAMES:
        print(">> Shape name {} not in dict, choosing randomly".format(name))
        name = "random"
    if name == "random":
        name = random.choice(SHAPENAMES)
    if not opts:
        opts = SHAPE_HELPERS[name](max_size)
    opts["labelme"] = False
    return SHAPEFNS[name](**opts), opts, name


def options_cube(max_size):
    return {"size": np.random.randint(MIN_SIZE, max_size + 1)}


def options_hollow_cube(max_size):
    opts = {}
    opts["size"] = np.random.randint(MIN_SIZE, max_size + 1)
    if opts["size"] < 5:
        opts["thickness"] = 1
    else:
        opts["thickness"] = np.random.randint(1, opts["size"] - 3)
    return opts


def options_rectanguloid(max_size):
    return {"size": np.random.randint(MIN_SIZE, max_size + 1, size=3)}


def options_hollow_rectanguloid(max_size):
    opts = {}
    opts["size"] = np.random.randint(MIN_SIZE, max_size + 1, size=3)
    ms = min(opts["size"])
    opts["thickness"] = np.random.randint(1, ms - 3 + 1)
    return opts


def options_sphere(max_size):
    min_r = MIN_SIZE // 2
    max_r = max_size // 2
    return {"radius": np.random.randint(min_r, max_r + 1)}


def options_spherical_shell(max_size):
    min_r = MIN_SIZE // 2
    max_r = max_size // 2
    opts = {}
    if max_r <= 5:
        opts["radius"] = np.random.randint(min_r, max_r + 1)
        opts["thickness"] = 1
    else:
        opts["radius"] = np.random.randint(5, max_r + 1)
        opts["thickness"] = np.random.randint(1, opts["radius"] - 3)
    return opts


# TODO: can we make this work??
def options_square_pyramid(max_size):
    min_r = MIN_SIZE
    max_r = max_size
    opts = {}
    opts["radius"] = np.random.randint(min_r, max_r + 1)
    opts["slope"] = np.random.rand() * 0.4 + 0.8
    fullheight = opts["radius"] * opts["slope"]
    opts["height"] = np.random.randint(0.5 * fullheight, fullheight)
    return opts


def options_square(max_size):
    return {"size": np.random.randint(MIN_SIZE, max_size + 1), "orient": sh.orientation3()}


def options_rectangle(max_size):
    return {"size": np.random.randint(MIN_SIZE, max_size + 1, size=2), "orient": sh.orientation3()}


def options_circle(max_size):
    min_r = MIN_SIZE // 2
    max_r = max_size // 2
    return {"radius": np.random.randint(min_r, max_r + 1), "orient": sh.orientation3()}


def options_disk(max_size):
    min_r = MIN_SIZE // 2
    max_r = max_size // 2
    return {"radius": np.random.randint(min_r, max_r + 1), "orient": sh.orientation3()}


def options_triangle(max_size):
    return {"size": np.random.randint(MIN_SIZE, max_size + 1), "orient": sh.orientation3()}


def options_dome(max_size):
    min_r = MIN_SIZE // 2
    max_r = max_size // 2
    return {"radius": np.random.randint(min_r, max_r + 1)}


# TODO: can we make this work
def options_arch(max_size):
    ms = max(MIN_SIZE + 1, max_size * 2 // 3)
    return {"size": np.random.randint(MIN_SIZE, ms), "distance": 2 * np.random.randint(2, 5) + 1}


def options_ellipsoid(max_size):
    # these sizes are actually radiuses
    min_r = MIN_SIZE // 2
    max_r = max_size // 2
    return {"size": np.random.randint(min_r, max_r + 1, size=3)}


def options_tower(max_size):
    return {"height": np.random.randint(3, max_size + 1), "base": np.random.randint(-4, 6)}


def options_empty(max_size):
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
def check_l1_dist(a, b, d):
    return abs(b[0] - a[0]) <= d[0] and abs(b[1] - a[1]) <= d[1] and abs(b[2] - a[2]) <= d[2]


def get_rectanguloid_subsegment(S, c, max_chunk=10):
    bounds, segment_sizes = su.get_bounds_and_sizes(S)
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


# Returns three tensors: 32x32x32 context, 8x8x8 segment, 1 target
class SegmentContextShapeData(torch.utils.data.Dataset):
    def __init__(self, nexamples=100000, context_side_length=32, seg_side_length=8, useid=False):
        self.c_sl = context_side_length
        self.s_sl = seg_side_length
        self.num_examples = nexamples
        self.useid = useid
        self.examples = []

    def _get_example(self):
        schem_sparse, seg = get_shape_segment(max_chunk=self.s_sl - 1, side_length=self.c_sl)
        seg_inds = set([i for i, use in enumerate(seg) if use])
        seg_sparse = [schem_sparse[i] for i in seg_inds]
        context_sparse = [b for i, b in enumerate(schem_sparse) if i not in seg_inds]
        return su.convert_sparse_context_seg_to_example(
            context_sparse, seg_sparse, self.c_sl, self.s_sl, self.useid
        )

    def __getitem__(self, index):
        return self._get_example()

    def __len__(self):
        return self.num_examples


def get_two_shape_sparse(c_sl, s_sl):
    max_s_size = random.randint(MIN_SIZE, s_sl)
    max_c_size = c_sl - 2 * max_s_size
    c_shape_sparse, _, nc = get_shape("random", max_c_size)
    s_shape_sparse, _, ns = get_shape("random", max_s_size)

    # move segment to (0,0,0) and bound size
    s_bounds, s_sizes = su.get_bounds_and_sizes(s_shape_sparse)
    seg_sparse, _ = su.shift_sparse_voxel_to_origin(s_shape_sparse)
    seg_sparse = [b for b in seg_sparse if all([i < 8 for i in b[0]])]
    s_bounds, s_sizes = su.get_bounds_and_sizes(seg_sparse)

    # ensure context isn't too big
    c_bounds, c_sizes = su.get_bounds_and_sizes(c_shape_sparse)
    total_sizes = [c + s * 2 for c, s in zip(c_sizes, s_sizes)]
    for i, size in enumerate(total_sizes):
        if size > 32:
            remove = size - 32
            remove_to = c_bounds[i * 2] + remove
            c_shape_sparse = [b for b in c_shape_sparse if b[0][i] >= remove_to]
    if len(c_shape_sparse) == 0:
        raise Exception("There should be something in c_shape_sparse {}".format(c_shape_sparse))
    c_bounds, c_sizes = su.get_bounds_and_sizes(c_shape_sparse)

    # shift context center to space center
    c_center = [sl // 2 for sl in c_sizes]
    c_space_center = [c_sl // 2 for _ in range(3)]
    sv = [sc - c for c, sc in zip(c_center, c_space_center)]
    context_sparse = [
        ((b[0][0] + sv[0], b[0][1] + sv[1], b[0][2] + sv[2]), b[1]) for b in c_shape_sparse
    ]

    return context_sparse, c_sizes, seg_sparse, s_sizes


def get_shape_dir_target(viewer_pos, dir_vec, c_sizes, s_sizes, c_sl, max_shift=0):
    c_space_center = [c_sl // 2 for _ in range(3)]
    c_half = [cs // 2 for cs in c_sizes]
    c_pos = [c_space_center[i] - c_half[i] for i in range(3)]
    target_coord, dim, dr = du.get_rotated_context_to_seg_origin(
        viewer_pos, dir_vec, c_pos, c_sizes, s_sizes
    )
    if any([t > c_sl - 1 or t < 0 for t in target_coord]):
        raise Exception("target coord:", target_coord)

    if max_shift > 0:
        # Shift by some precalculated amount and turn into a target
        shift_constraint = c_space_center[dim] - c_half[dim] - s_sizes[dim] - 2
        shift_by = random.randint(0, min(max(shift_constraint, 0), max_shift))
        target_coord[dim] += dr * shift_by
    target = su.coord_to_index(target_coord.tolist(), c_sl)
    return torch.tensor(target, dtype=torch.int)


# Returns a 32x32x32 context, 8x8x8 segment, 6 viewer, 1 direction, 1 target
class SegmentContextShapeDirData(torch.utils.data.Dataset):
    def __init__(
        self, nexamples=100000, context_side_length=32, seg_side_length=8, useid=False, max_shift=0
    ):
        self.c_sl = context_side_length
        self.s_sl = seg_side_length
        self.num_examples = nexamples
        self.useid = useid
        self.examples = []
        self.max_shift = max_shift

    def _get_example(self):
        # note that seg_sparse is not in target location
        context_sparse, c_sizes, seg_sparse, s_sizes = get_two_shape_sparse(self.c_sl, self.s_sl)
        viewer_pos, viewer_look = du.get_random_viewer_info(self.c_sl)
        dir_vec = du.random_dir_vec_tensor()
        target = get_shape_dir_target(
            viewer_pos, dir_vec, c_sizes, s_sizes, self.c_sl, self.max_shift
        )
        context = su.get_dense_array_from_sl(context_sparse, self.c_sl, self.useid)
        seg = su.get_dense_array_from_sl(seg_sparse, self.s_sl, self.useid)
        return {
            "context": torch.from_numpy(context),
            "seg": torch.from_numpy(seg),
            "target": target,
            "viewer_pos": viewer_pos,
            "dir_vec": dir_vec,
        }

    def __getitem__(self, index):
        return self._get_example()

    def __len__(self):
        return self.num_examples


if __name__ == "__main__":
    from visualization_utils import GeoscorerDatasetVisualizer

    dataset = SegmentContextShapeDirData(nexamples=3)

    vis = GeoscorerDatasetVisualizer(dataset)
    for n in range(len(dataset)):
        vis.visualize()
