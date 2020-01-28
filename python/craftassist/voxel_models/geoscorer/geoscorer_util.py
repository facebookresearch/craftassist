"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import numpy as np
import random
from datetime import datetime
import sys
import argparse
import torch
import os
from inspect import currentframe, getframeinfo

GEOSCORER_DIR = os.path.dirname(os.path.realpath(__file__))
CRAFTASSIST_DIR = os.path.join(GEOSCORER_DIR, "../")
sys.path.append(CRAFTASSIST_DIR)

from shapes import get_bounds


def pretty_log(log_string):
    cf = currentframe().f_back
    filename = getframeinfo(cf).filename.split("/")[-1]
    print(
        "{} {}:{} {}".format(
            datetime.now().strftime("%m/%d/%Y %H:%M:%S"), filename, cf.f_lineno, log_string
        )
    )
    sys.stdout.flush()


## Train Fxns ##


def get_base_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=1, help="0 for cpu")
    parser.add_argument("--batchsize", type=int, default=64, help="batchsize")
    parser.add_argument("--dataset", default="shapes", help="shapes/segments/both")
    parser.add_argument(
        "--epochsize", type=int, default=1000, help="number of examples in an epoch"
    )
    parser.add_argument("--nepoch", type=int, default=1000, help="number of epochs")
    parser.add_argument("--context_sidelength", type=int, default=32, help="size of cube")
    parser.add_argument("--hidden_dim", type=int, default=64, help="size of hidden dim")
    parser.add_argument("--num_layers", type=int, default=3, help="num layers")
    parser.add_argument(
        "--blockid_embedding_dim", type=int, default=8, help="size of blockid embedding"
    )
    parser.add_argument(
        "--num_words", type=int, default=256, help="number of words for the blockid embeds"
    )
    parser.add_argument("--lr", type=float, default=0.1, help="step size for net")
    parser.add_argument(
        "--optim", type=str, default="adagrad", help="optim type to use (adagrad|sgd|adam)"
    )
    parser.add_argument("--momentum", type=float, default=0.0, help="momentum")
    parser.add_argument("--checkpoint", default="", help="where to save model")
    parser.add_argument("--num_workers", type=int, default=4, help="number of dataloader workers")
    return parser


def add_dataset_flags(parser):
    parser.add_argument(
        "--dataset_ratios", type=str, default="shape:1.0", help="comma separated name:prob"
    )
    parser.add_argument("--useid", type=bool, default=False, help="use blockid")
    parser.add_argument("--fixed_cube_size", type=int, default=None, help="fixed_cube_size")
    parser.add_argument("--fixed_center", type=bool, default=False, help="fixed_center")
    parser.add_argument(
        "--min_seg_size", type=int, default=6, help="min seg size for seg data type"
    )
    parser.add_argument(
        "--use_saved_data",
        type=bool,
        default=False,
        help="use preparsed data for this min_seg_size",
    )


def add_directional_flags(parser):
    parser.add_argument("--spatial_embedding_dim", type=int, default=8, help="size of spatial emb")
    parser.add_argument("--output_embedding_dim", type=int, default=8, help="size of output emb")
    parser.add_argument(
        "--seg_direction_net", type=bool, default=False, help="use segdirnet module"
    )
    parser.add_argument(
        "--seg_use_viewer_pos", type=bool, default=False, help="use viewer pos in seg"
    )
    parser.add_argument(
        "--seg_use_viewer_look", type=bool, default=False, help="use viewer look in seg"
    )
    parser.add_argument(
        "--seg_use_direction", type=bool, default=False, help="use direction in seg"
    )
    parser.add_argument("--num_seg_dir_layers", type=int, default=3, help="num segdir net layers")
    parser.add_argument(
        "--cont_use_direction", type=bool, default=False, help="use direction in context"
    )
    parser.add_argument(
        "--cont_use_xyz_from_viewer_look",
        type=bool,
        default=False,
        help="use xyz position relative to viewer look in context emb",
    )


def get_dataloader(dataset, opts, collate_fxn):
    def init_fn(wid):
        np.random.seed(torch.initial_seed() % (2 ** 32))

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=opts["batchsize"],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=opts["num_workers"],
        worker_init_fn=init_fn,
        collate_fn=collate_fxn,
    )


def to_cuda(list_modules):
    for m in list_modules:
        m.cuda()


def multitensor_collate_fxn(x):
    """
    Takes a list of BATCHSIZE lists of tensors of length D.
    Returns a list of length D of batched tensors.
    """
    num_tensors_to_batch = len(x[0])

    regroup_tensors = [[] for i in range(num_tensors_to_batch)]
    for t_list in x:
        for i, t in enumerate(t_list):
            regroup_tensors[i].append(t.unsqueeze(0))
    batched_tensors = [torch.cat(tl) for tl in regroup_tensors]
    return batched_tensors


## 3D Utils ##


def get_side_lengths(bounds):
    """
    Bounds should be a list of [min_x, max_x, min_y, max_y, min_z, max_z].
    Returns a list of the side lengths.
    """
    return [x + 1 for x in (bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])]


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


def shift_subsegment_corner(S):
    """
    Takes a segment, described as a list of tuples of the form:
        ((x, y, z), (block_id, ?))
    Returns the segment in the same form, shifted to the origin, and the shift vec
    """
    bounds = get_bounds(S)
    shift_zero_vec = [-bounds[0], -bounds[2], -bounds[4]]
    new_S = []
    for s in S:
        new_S.append((tuple([sum(x) for x in zip(s[0], shift_zero_vec)]), s[1]))
    return new_S, shift_zero_vec


def subset_and_scale_3d(init_array, mins, maxs, scale=1):
    return scale * init_array[mins[0] : maxs[0], mins[1] : maxs[1], mins[2] : maxs[2]]


def combine_seg_context(seg, context, seg_shift, seg_mult=1):
    completed_context = context.clone()

    # Calculate the region to copy over, sometimes the segment
    #   falls outside the range of the context bounding box
    c_mins = [int(i) for i in seg_shift]
    c_maxs = [int(min(ss + 8, 32)) for ss in seg_shift]
    s_mins = [0 for i in range(3)]
    # If the edge of the segment goes past the edge of the context (ss + 8 > 32),
    #   remove the extra from the segment.
    s_maxs = [int(8 - max(0, (ss + 8) - 32)) for ss in seg_shift]

    seg_to_add = subset_and_scale_3d(seg, s_mins, s_maxs, seg_mult)
    context_subset = subset_and_scale_3d(completed_context, c_mins, c_maxs, 1)
    completed_context[c_mins[0] : c_maxs[0], c_mins[1] : c_maxs[1], c_mins[2] : c_maxs[2]] = (
        seg_to_add + context_subset
    )
    return completed_context


def get_vector(start, end):
    return end - start


def get_random_viewer_info(sl):
    viewer_pos = torch.tensor(random_int_triple(0, sl - 1))
    viewer_look = torch.tensor(random_int_triple(0, sl - 1))

    if viewer_pos.eq(viewer_look).sum() == viewer_pos.size(0):
        if viewer_look[0] < sl + 1:
            viewer_look[0] += 1
        else:
            viewer_look[0] -= 1
    return viewer_pos, viewer_look


def b_greater_than_a(a, b):
    if a == b:
        return 0
    return 1 if b > a else -1


def shift_block(b, s):
    return tuple((tuple((b[0][0] + s[0], b[0][1] + s[1], b[0][2] + s[2])), b[1]))


def rotate_block(b, c, r):
    """ rotates the block b around the point c by 90*r degrees
    in the xz plane.  r should be 1 or -1."""
    # TODO add a reflection
    c = np.array(c)
    p = np.add(b[0], -c)
    x = p[0]
    z = p[2]
    if r == -1:
        p[0] = z
        p[2] = -x
    else:
        p[0] = -z
        p[2] = x
    return (tuple(p + c), b[1])


def random_int_triple(minval, maxval):
    t = [
        random.randint(minval, maxval),
        random.randint(minval, maxval),
        random.randint(minval, maxval),
    ]
    return t


def check_inrange(x, minval, maxval):
    """inclusive check"""
    return all([v >= minval for v in x]) and all([v <= maxval for v in x])


def normalize(batched_vector):
    vec = batched_vector.double()
    norm = torch.norm(vec, dim=1)
    # Set norm to 1 if it's 0
    norm = norm + norm.eq(0).double()
    expanded_norm = norm.unsqueeze(1).expand(-1, vec.size()[1])
    return torch.div(vec, expanded_norm)


def get_rotation_matrix(viewer_pos, viewer_look):
    # VP, VL: N x 3, VP_to_VL: N x 3
    vp_to_vl = get_vector(viewer_pos, viewer_look)[:, :2]
    nlook_vec = normalize(vp_to_vl)
    nly = nlook_vec[:, 1]
    # Nlx necessary to correct for the range of acrcos
    nlx = nlook_vec[:, 0]
    nlx = nlx.gt(0).double() - nlx.lt(0).double() - nlx.eq(0).double()

    # Take care of nans created by raising 0 to a power
    # and then masking the sin theta to 0 as intended
    base = 1 - nly * nly
    nan_mask = torch.isnan(torch.pow(base, 0.5)).double()
    base = base + nan_mask
    sin_theta = nlx * nan_mask.eq(0).double() * torch.pow(base, 0.5)

    nly = nly.unsqueeze(1)
    sin_theta = sin_theta.unsqueeze(1)
    rm_pt1 = torch.cat([nly, sin_theta], 1).unsqueeze(1)
    rm_pt2 = torch.cat([-sin_theta, nly], 1).unsqueeze(1)
    rm = torch.cat([rm_pt1, rm_pt2], 1)
    return rm


def rotate_x_y(coord, rotation_matrix):
    return torch.mm(coord.unsqueeze(0), rotation_matrix).squeeze(0)


def float_equals(a, b, epsilon):
    return True if abs(a - b) < epsilon else False


def get_argmax_list(vals, epsilon, minlist=False, maxlen=None):
    mult = -1 if minlist else 1
    max_ind = []
    for i, v in enumerate(vals):
        if not max_ind or float_equals(max_ind[0][1], v, epsilon):
            if maxlen and len(max_ind) == maxlen:
                continue
            max_ind.append((i, v))
        elif mult * (v - max_ind[0][1]) > 0:
            max_ind = [(i, v)]
    return max_ind


def get_firstmax(vals, epsilon, minlist=False):
    return get_argmax_list(vals, epsilon, minlist, 1)[0]


# N -> batch size in training
# D -> num target coord per element
# Viewer pos, viewer_look are N x 3 tensors
# Batched target coords is a N x D x 3 tensor
# Output is a N x D x 3 tensor
def get_xyz_viewer_look_coords_batched(viewer_pos, viewer_look, batched_target_coords):
    # First verify the sizing and unsqueeze if necessary
    btc_sizes = batched_target_coords.size()
    vp_sizes = viewer_pos.size()
    vl_sizes = viewer_look.size()
    if len(btc_sizes) > 3 or len(vp_sizes) > 2 or len(vl_sizes) > 2:
        raise Exception("One input has too many dimensions")
    if btc_sizes[-1] != 3 or vp_sizes[-1] != 3 or vl_sizes[-1] != 3:
        raise Exception("The last dimension of all inputs should be size 3")
    if len(btc_sizes) < 3:
        for i in range(3 - len(btc_sizes)):
            batched_target_coords = batched_target_coords.unsqueeze(0)
    if len(vp_sizes) == 1:
        viewer_pos = viewer_pos.unsqueeze(0)
    if len(vl_sizes) == 1:
        viewer_look = viewer_look.unsqueeze(0)
    n = batched_target_coords.size()[0]
    d = batched_target_coords.size()[1]

    # Handle xy and z separately
    # XY = N X D x 2
    xy = batched_target_coords[:, :, 0:2].double()
    # Z = N x D x 1
    z = batched_target_coords[:, :, 2].unsqueeze(2).double()

    ##  XY
    # Shift such that viewer pos is the origin

    # VPXY, VLXY: N x 2
    vpxy = viewer_pos.double()[:, 0:2]
    vlxy = viewer_look.double()[:, 0:2]
    vpxy_to_vlxy = vlxy - vpxy
    # VPXY to XY: N x D x 2
    vpxy_to_xy = xy - vpxy.unsqueeze(1).expand(n, d, -1)

    # Rotate them around the viewer position such that a normalized
    # viewer look vector would be (0, 1)
    # Rotation_matrix: N x 2 x 2
    rotation_matrix = get_rotation_matrix(viewer_pos, viewer_look)

    # N x 1 x 2 mm N x 2 x 2 ==> N x 1 x 2 ==> N x 2
    r_vpxy_to_vlxy = torch.bmm(vpxy_to_vlxy.unsqueeze(1), rotation_matrix).unsqueeze(1)

    # RM: N x 2 x 2 ==> N x D x 2 x 2
    expanded_rm = rotation_matrix.unsqueeze(1).expand(n, d, 2, 2).contiguous().view(-1, 2, 2)
    # N x D x 2 ==> N*D x 1 x 2 mm N*D x 2 x 2 ==> N*D x 1 x 2 ==> N x D x 2
    reshape_vpxy_to_xy = vpxy_to_xy.contiguous().view(-1, 1, 2)
    r_vpxy_to_xy = torch.bmm(reshape_vpxy_to_xy, expanded_rm).contiguous().view(n, d, 2)

    # N x D x 2
    # Get the xy position in this rotated coord system with rvl as the origin
    rvl_to_rxy = r_vpxy_to_xy - r_vpxy_to_vlxy.squeeze(1).expand(n, d, 2)

    ## Z
    # VLZ = N x 1
    vlz = viewer_look.double()[:, 2]
    # Z = N x D x 1
    diffz = z - vlz.view(-1, 1, 1).expand(n, d, -1)

    ## Combine
    # rvl_to_rxy: N x D x 2, diffz: N x D x 1
    new_xyz = torch.cat([rvl_to_rxy, diffz], 2)
    return new_xyz


def get_dir_dist(viewer_pos, viewer_look, batched_target_coords):
    if len(batched_target_coords.size()) == 1:
        batched_target_coords = batched_target_coords.unsqueeze(0)
    xyz = get_xyz_viewer_look_coords_batched(viewer_pos, viewer_look, batched_target_coords)
    dist = xyz.abs()
    direction = xyz.gt(0).double() - xyz.lt(0).double()
    return direction, dist


def get_sampled_direction_vec(viewer_pos, viewer_look, target_coord):
    directions, dists = get_dir_dist(viewer_pos, viewer_look, target_coord)
    dists = dists.squeeze()
    directions = directions.squeeze()
    ndists = dists / sum(dists)
    dim = np.random.choice(3, p=ndists)
    direction = directions[dim].item()
    dim_l = [(0 if i == dim else 1) for i in range(3)]
    dir_l = [0, 1] if direction == -1 else [1, 0]
    return torch.tensor(dim_l + dir_l, dtype=torch.long)


def get_max_direction_vec(viewer_pos, viewer_look, target_coord):
    directions, dists = get_dir_dist(viewer_pos, viewer_look, target_coord)
    dists = dists.squeeze()
    directions = directions.squeeze()
    ndists = dists / sum(dists)
    dim = np.argmax(ndists)
    direction = directions[dim].item()
    dim_l = [(0 if i == dim else 1) for i in range(3)]
    dir_l = [0, 1] if direction == -1 else [1, 0]
    return torch.tensor(dim_l + dir_l, dtype=torch.long)


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


def center_of_mass(S, seg=None):
    seg = seg or [True for i in S]
    if len(S[0]) == 2:
        m = list(np.round(np.mean([S[i][0] for i in range(len(S)) if seg[i]], axis=0)))
    else:
        m = list(np.round(np.mean([S[i] for i in range(len(S)) if seg[i]], axis=0)))
    return [int(i) for i in m]


def check_l1_dist(a, b, d):
    return abs(b[0] - a[0]) <= d[0] and abs(b[1] - a[1]) <= d[1] and abs(b[2] - a[2]) <= d[2]


def sparsify_segment(seg, context):
    seg_sparse = []
    for i, use in enumerate(seg):
        if use:
            seg_sparse.append(context[i])
    return seg_sparse


def get_dense_array_from_sl(sparse_shape, sl, useid):
    center = [sl // 2, sl // 2, sl // 2]
    shape_dense, _ = np.asarray(densify(sparse_shape, [sl, sl, sl], center=center, useid=useid))
    return shape_dense


def convert_sparse_context_seg_to_example(
    context_sparse, seg_sparse, c_sl, s_sl, useid, vis=False
):
    context_dense = get_dense_array_from_sl(context_sparse, c_sl, useid)
    seg_dense_uncentered = get_dense_array_from_sl(seg_sparse, c_sl, useid)

    # For visualization
    if vis:
        context_dense = context_dense + seg_dense_uncentered
    else:
        context_dense = context_dense - seg_dense_uncentered

    shifted_seg_sparse, shift_vec = shift_subsegment_corner(seg_sparse)
    seg_dense_centered = get_dense_array_from_sl(shifted_seg_sparse, s_sl, useid)
    target_coord = [-x for x in shift_vec]
    target_index = coord_to_index(target_coord, c_sl)
    return [
        torch.from_numpy(context_dense),
        torch.from_numpy(seg_dense_centered),
        torch.tensor([target_index]),
    ]


############################################################################
# For these "S" is a list of blocks in ((x,y,z),(id, meta)) format
# the segment is a list of the same length as S with either True or False
# at each entry marking whether that block is in the segment
# each outputs a list of blocks in ((x,y,z),(id, meta)) format


def shift_negative_vec(S, segment, vec, args):
    N = []
    for s in range(len(segment)):
        if not segment[s]:
            new_coords = tuple(np.add(S[s][0], vec))
            N.append([new_coords, S[s][1]])
        else:
            if "seg_id" in args:
                N.append([S[s][0], (args["seg_id"], S[s][1][1])])
            else:
                N.append(S[s])
    return N


def shift_negative(S, segment, args):
    shift_max = args["shift_max"]
    """takes the blocks not in the sgement and shifts them randomly"""
    shift_vec = random_int_triple(-shift_max, shift_max)
    return shift_negative_vec(S, segment, shift_vec, args)


def rotate_negative(S, segment, args):
    c = center_of_mass(S, seg=segment)
    r = random.choice([1, -1])
    return [rotate_block(S[i], c, r) if segment[i] else S[i] for i in range(len(S))]


def replace_negative(S, segment, args):
    data = args["data"]
    oseg, oS = data.get_positive()
    c_pos = center_of_mass(S, seg=segment)
    c_neg = center_of_mass(oS, seg=oseg)
    offset = np.add(c_pos, -np.array(c_neg))
    N = [S[i] for i in range(len(S)) if not segment[i]]
    return N + [shift_block(oS[i], offset) for i in range(len(oS)) if oseg[i]]


class NegativeSampler:
    def __init__(self, dataloader, shift_max=10, ntype_probs=[0.6, 0.2, 0.2]):
        #        self.data_prob = [x['prob'] for x in dataloaders.values()]
        #        self.dataloaders = [x['data'] for x in dataloaders.values()]
        self.dataloader = dataloader
        self.shift_max = shift_max
        self.ntype_probs = ntype_probs
        self.negative_samplers = [shift_negative, rotate_negative, replace_negative]

    def build_negative(self, S, segment):
        negative_fn = np.random.choice(self.negative_samplers, p=self.ntype_probs)
        return negative_fn(S, segment, {"shift_max": self.shift_max, "data": self.dataloader})
