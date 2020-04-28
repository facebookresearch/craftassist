"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import numpy as np
import torch
import random


def get_viewer_look(c_sl):
    return torch.tensor([c_sl // 2 for _ in range(3)])


def get_random_viewer_info(sl):
    viewer_pos = torch.tensor(np.random.randint(0, sl, 3))
    viewer_look = get_viewer_look(sl)

    if viewer_pos.eq(viewer_look).sum() == viewer_pos.size(0):
        if viewer_pos[0] < sl + 1:
            viewer_pos[0] += 1
        else:
            viewer_pos[0] -= 1
    return viewer_pos, viewer_look


def get_vector(start, end):
    return end - start


def dim_to_vec(dim):
    return [(1 if i == dim else 0) for i in range(3)]


def dir_vec_to_dim(dir_vec):
    for i in range(3):
        if dir_vec[i] == 1:
            return i
    raise Exception("dir vec has no dimension")


def dr_to_vec(dr):
    return [1, 0] if dr == 1 else [0, 1]


def dir_vec_to_dr(dir_vec):
    if dir_vec[3] == 1:
        return 1
    elif dir_vec[4] == 1:
        return -1
    else:
        raise Exception("dir vec has no direction")


def dim_dir_to_dir_tensor(dim, dr):
    dim_l = dim_to_vec(dim)
    dir_l = dr_to_vec(dr)
    return torch.tensor(dim_l + dir_l, dtype=torch.long)


def dir_vec_to_dim_dir(dir_vec):
    dim = dir_vec_to_dim(dir_vec)
    dr = dir_vec_to_dr(dir_vec)
    return dim, dr


def random_dir_vec_tensor():
    dim = random.choice([0, 1, 2])
    dr = random.choice([-1, 1])
    return dim_dir_to_dir_tensor(dim, dr)


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


def get_argmax_list(vals, epsilon=0.0001, minlist=False, maxlen=None):
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


def get_firstmax(vals, epsilon=0.0001, minlist=False):
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
    return dim_dir_to_dir_tensor(dim, direction)


def get_max_direction_vec(viewer_pos, viewer_look, target_coord):
    directions, dists = get_dir_dist(viewer_pos, viewer_look, target_coord)
    dists = dists.squeeze()
    directions = directions.squeeze()
    ndists = dists / sum(dists)
    dim = np.argmax(ndists)
    direction = directions[dim].item()
    return dim_dir_to_dir_tensor(dim, direction)


def convert_origin_to_center(origin, sizes):
    half = [s // 2 for s in sizes]
    return [origin[i] + half[i] for i in range(3)]


def convert_center_to_origin(center, sizes):
    half = [s // 2 for s in sizes]
    return [center[i] - half[i] for i in range(3)]


def get_rotated_context_to_seg_origin(viewer_pos, dir_vec, c_pos, c_sizes, s_sizes):
    c_half = [sl // 2 for sl in c_sizes]
    s_half = [sl // 2 for sl in s_sizes]
    c_center = [c_pos[i] + c_half[i] for i in range(3)]
    dim, dr = dir_vec_to_dim_dir(dir_vec)

    # These account for the sl // 2 discretization, apply in positive direction
    # TODO: there must be a way to only use one of these
    c_offset_even = [1 if c % 2 == 0 else 0 for c in c_sizes]
    c_offset_odd = [c % 2 for c in c_sizes]

    # For above below, directly attach in that dir
    if dim == 2:
        touch_p = [i for i in c_center]
        if dr == -1:
            touch_p[2] -= c_half[2]
            target_coord = [
                touch_p[0] - s_half[0],
                touch_p[1] - s_half[1],
                touch_p[2] - s_sizes[2],
            ]
        else:
            touch_p[2] += c_half[2] + c_offset_odd[2]
            target_coord = [touch_p[0] - s_half[0], touch_p[1] - s_half[1], touch_p[2]]
        return torch.tensor(target_coord, dtype=torch.int), dim, dr

    # Find the 4 possible positions
    c_shift = [c_half[i] + 1 for i in range(2)]
    possible_touch_points = []
    possible_targets = []
    shift_dims = []
    shift_dirs = []
    for sdim in [0, 1]:
        for sdr in [1, -1]:
            shift_dims.append(sdim)
            shift_dirs.append(sdr)
            tp = [p for p in c_center]
            tp[sdim] += sdr * c_shift[sdim]
            if sdr > 0:
                tp[sdim] -= c_offset_even[sdim]
            possible_touch_points.append(torch.tensor(tp, dtype=torch.float))

            t = [p for p in tp]
            for d in range(3):
                if d == sdim:
                    if sdr < 0:
                        t[d] -= s_sizes[d] - 1
                else:
                    t[d] -= s_half[d]
            possible_targets.append(torch.tensor(t, dtype=torch.float))

    # Chooose the best touch point based on rotation
    c_center_t = torch.tensor(c_center, dtype=torch.float)
    c_to_ts = [get_vector(c_center_t[:2], t[:2]) for t in possible_touch_points]
    rotation_matrix = get_rotation_matrix(
        viewer_pos.unsqueeze(0), c_center_t.unsqueeze(0)
    ).squeeze(0)
    c_to_t_rotated = [rotate_x_y(c_to_t.double(), rotation_matrix) for c_to_t in c_to_ts]

    vals = [v[dim] for v in c_to_t_rotated]
    if dr == 1:
        max_ind, dist = get_firstmax(vals)
    else:
        max_ind, dist = get_firstmax(vals, minlist=True)

    target_coord = possible_targets[max_ind]
    return target_coord.int(), shift_dims[max_ind], shift_dirs[max_ind]
