import numpy as np
import random
import os
import mc_block_ids as mbi


def ground_height(blocks):
    """
    This function is used in render_one_block_change.py. The recorded changed block is
    after removing the ground. So we need to keep it consistent.
    """
    dirt_pct = np.mean(np.mean(blocks[:, :, :, 0] == 2, axis=1), axis=1)
    if (dirt_pct > 0.25).any():
        return np.argmax(dirt_pct)
    return None


def get_voxel_center_cube(loc, schematic, size):
    """
    Given a cube size, we would like to cut a cube centered at loc
    from schematic. The cube by default is all zeros.

    This cube is a standard unit cube for training or inference. Multiple such
    cubes constitute a minibatch.

    'schematic' has x-y-z-b, 'loc' has x-y-z, and 'cube' has x-y-z

    This function could be called online by the agent. So for efficiency,
    we need to use array broadcasting instead of element-wise indexing.
    """

    def clamp(a, ma, Ma):
        return min(max(ma, a), Ma)

    assert size % 2 == 1, "Only support odd sizes!"
    ## embedding layer requires long type
    cube = np.zeros((size, size, size, 1), dtype="float32")
    x, y, z = loc
    offset = size // 2
    X, Y, Z, _ = schematic.shape

    My = clamp(y + offset, 0, Y - 1)
    my = clamp(y - offset, 0, Y - 1)
    Mz = clamp(z + offset, 0, Z - 1)
    mz = clamp(z - offset, 0, Z - 1)
    Mx = clamp(x + offset, 0, X - 1)
    mx = clamp(x - offset, 0, X - 1)

    cmy = my - y + offset
    cmz = mz - z + offset
    cmx = mx - x + offset
    cMy = My - y + offset
    cMz = Mz - z + offset
    cMx = Mx - x + offset

    ## only take block id [0] and ignore meta id [1]
    cube[cmx : cMx + 1, cmy : cMy + 1, cmz : cMz + 1, 0] = schematic[
        mx : Mx + 1, my : My + 1, mz : Mz + 1, 0
    ]

    ## fill dirt or grass
    if cmy > 0:
        cube[:, :cmy, :, 0] = random.choice([2, 3])

    cube = mbi.check_block_id()(cube)
    return cube


def get_batch_voxel_center_cubes(locs, schematic, size):
    """
    Given a list of locations, this function assembles a minibatch by cutting a cube of size
    around each location from schematic.
    """
    return np.array([get_voxel_center_cube(loc, schematic, size) for loc in locs])
