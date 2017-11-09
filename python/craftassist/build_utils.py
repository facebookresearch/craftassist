import csv
import logging
import numpy as np

from block_data import BORING_BLOCKS, PASSABLE_BLOCKS
from search import depth_first_search
import util

MAX_RADIUS = 20
SHAPENET_PATH = ""  # add path to shapenet csv here


def read_shapenet_csv(csvpath=SHAPENET_PATH + "metadata.csv"):
    keyword_to_id = {}
    id_to_keyword = {}
    with open(csvpath, "r") as cfile:
        cvs_metadata = csv.reader(cfile)
        next(cvs_metadata)
        for l in cvs_metadata:
            bin_id = l[0][4:]
            keywords = l[3].split(",")
            if keywords[0] == "":
                continue
            id_to_keyword[bin_id] = keywords
            for k in keywords:
                if keyword_to_id.get(k) is None:
                    keyword_to_id[k] = []
                else:
                    keyword_to_id[k].append(bin_id)
    return keyword_to_id, id_to_keyword


def to_relative_pos(block_list):
    """Convert absolute block positions to their relative positions

    Find the "origin", i.e. the minimum (x, y, z), and subtract this from all
    block positions.

    Args:
    - block_list: a list of ((x,y,z), (id, meta))

    Returns:
    - a block list where positions are shifted by `origin`
    - `origin`, the (x, y, z) offset by which the positions were shifted
    """
    try:
        locs, idms = zip(*block_list)
    except ValueError:
        raise ValueError("to_relative_pos invalid input: {}".format(block_list))

    locs = np.array([loc for (loc, idm) in block_list])
    origin = np.min(locs, axis=0)
    locs -= origin
    S = [(tuple(loc), idm) for (loc, idm) in zip(locs, idms)]
    if type(block_list) is not list:
        S = tuple(S)
    if type(block_list) is frozenset:
        S = frozenset(S)
    return S, origin


def all_nearby_objects(get_blocks, pos):
    """Return a list of connected components near pos.

    Each component is a list of ((x, y, z), (id, meta))

    i.e. this function returns list[list[((x, y, z), (id, meta))]]
    """
    pos = np.round(pos).astype("int32")
    mask, off, blocks = all_close_interesting_blocks(get_blocks, pos)
    components = connected_components(mask)
    logging.debug("all_nearby_objects found {} objects near {}".format(len(components), pos))
    xyzbms = [
        [((c[2] + off[2], c[0] + off[0], c[1] + off[1]), tuple(blocks[c])) for c in component_yzxs]
        for component_yzxs in components
    ]
    return xyzbms


# these should all be in perception...
def closest_nearby_object(get_blocks, pos):
    """Find the closest interesting object to pos

    Returns a list of ((x,y,z), (id, meta)), or None if no interesting objects are nearby
    """
    objects = all_nearby_objects(get_blocks, pos)
    if len(objects) == 0:
        return None
    centroids = [np.mean([pos for (pos, idm) in obj], axis=0) for obj in objects]
    dists = [util.manhat_dist(c, pos) for c in centroids]
    return objects[np.argmin(dists)]


# should be in perception
def all_close_interesting_blocks(get_blocks, pos, max_radius=MAX_RADIUS):
    mx, my, mz = pos[0] - max_radius, pos[1] - max_radius, pos[2] - max_radius
    Mx, My, Mz = pos[0] + max_radius, pos[1] + max_radius, pos[2] + max_radius

    yzxb = get_blocks(mx, Mx, my, My, mz, Mz)
    relpos = pos - [mx, my, mz]
    mask = accessible_interesting_blocks(yzxb[:, :, :, 0], relpos)
    return mask, (my, mz, mx), yzxb


# and this one
def accessible_interesting_blocks(blocks, pos):
    """Return a boolean mask of blocks that are accessible-interesting from pos.

    A block b is accessible-interesting if it is
    1. interesting, AND
    2. there exists a path from pos to b through only passable or interesting blocks
    """
    passable = np.isin(blocks, PASSABLE_BLOCKS)
    interesting = np.isin(blocks, BORING_BLOCKS, invert=True)
    passable_or_interesting = passable | interesting
    X = np.zeros_like(passable)

    def _fn(p):
        if passable_or_interesting[p]:
            X[p] = True
            return True
        return False

    depth_first_search(blocks, pos, _fn)
    return X & interesting


# and this one
def find_closest_component(mask, relpos):
    """Find the connected component of nonzeros that is closest to loc

    Args:
    - mask is a 3d array
    - relpos is a relative position in the mask, with the same ordering

    Returns: a list of indices of the closest connected component, or None
    """
    components = connected_components(mask)
    if len(components) == 0:
        return None
    centroids = [np.mean(cs, axis=0) for cs in components]
    dists = [util.manhat_dist(c, relpos) for c in centroids]
    return components[np.argmin(dists)]


# and this one
def connected_components(X):
    """Find all connected nonzero components in a 3d array X

    Returns a list of lists of indices of connected components
    """
    visited = np.zeros_like(X, dtype="bool")
    components = []
    current_component = set()

    def _fn(p):
        if X[p]:
            current_component.add(p)
            return True

    for i in range(visited.shape[0]):
        for j in range(visited.shape[1]):
            for k in range(visited.shape[2]):
                if visited[i, j, k]:
                    continue
                visited[i, j, k] = True
                if not X[i, j, k]:
                    continue
                # found a new component
                pos = (i, j, k)
                visited |= depth_first_search(X, pos, _fn, util.diag_adjacent)
                components.append(list(current_component))
                current_component.clear()

    return components


def blocks_list_to_npy(blocks, xyz=False):
    xyzbm = np.array([(x, y, z, b, m) for ((x, y, z), (b, m)) in blocks])
    mx, my, mz = np.min(xyzbm[:, :3], axis=0)
    Mx, My, Mz = np.max(xyzbm[:, :3], axis=0)

    npy = np.zeros((My - my + 1, Mz - mz + 1, Mx - mx + 1, 2), dtype="uint8")

    for x, y, z, b, m in xyzbm:
        npy[y - my, z - mz, x - mx] = (b, m)

    offsets = (my, mz, mx)

    if xyz:
        npy = np.swapaxes(np.swapaxes(npy, 1, 2), 0, 1)
        offsets = (mx, my, mz)

    return npy, offsets


def npy_to_blocks_list(npy, origin=(0, 0, 0)):
    blocks = []
    sy, sz, sx, _ = npy.shape
    for ry in range(sy):
        for rz in range(sz):
            for rx in range(sx):
                idm = tuple(npy[ry, rz, rx, :])
                if idm[0] == 0:
                    continue
                xyz = tuple(np.array([rx, ry, rz]) + origin)
                blocks.append((xyz, idm))
    return blocks


def blocks_list_add_offset(blocks, origin):
    """Offset all blocks in block list by a constant xyz

    Args:
      blocks: a list[(xyz, idm)]
      origin: xyz

    Returns list[(xyz, idm)]
    """
    ox, oy, oz = origin
    return [((x + ox, y + oy, z + oz), idm) for ((x, y, z), idm) in blocks]
