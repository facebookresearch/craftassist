import torch
import numpy as np
import mc_block_ids as mbi


def nn_resize_cube(cube, size):
    """
    Resize cube of any shape to sizexsizexsize using nearest neighbors
    The cube can be either x-y-z or c-x-y-z
    """
    ori_size = cube.size()
    if len(ori_size) == 3:
        cube = cube.unsqueeze(0)
    C, X, Y, Z = cube.size()

    def one_index(x, y, z):
        return x * Y * Z + y * Z + z

    xs = [int(X * i / size) for i in range(size)]
    ys = [int(Y * i / size) for i in range(size)]
    zs = [int(Z * i / size) for i in range(size)]
    cube = cube.view(C, -1)
    idx = torch.tensor([one_index(x, y, z) for x in xs for y in ys for z in zs])
    if idx.device != cube.device:
        idx = idx.to(cube.device)
    resized_cube = torch.index_select(cube, 1, idx)
    resized_cube = resized_cube.view(C, size, size, size)
    if len(ori_size) == 3:
        resized_cube = resized_cube.squeeze(0)

    return resized_cube


def clamp(a, ma, Ma):
    return min(max(ma, a), Ma)


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
    if not isinstance(size, tuple):
        size = (size, size, size)
    assert len(size) == 3

    ## embedding layer requires long type
    cube = np.zeros((size[0], size[1], size[2], 1), dtype="int64")
    x, y, z = loc
    offset_x_l = size[0] // 2
    offset_x_h = size[0] // 2 if size[0] % 2 == 1 else size[0] // 2 - 1
    offset_y_l = size[1] // 2
    offset_y_h = size[1] // 2 if size[1] % 2 == 1 else size[1] // 2 - 1
    offset_z_l = size[2] // 2
    offset_z_h = size[2] // 2 if size[2] % 2 == 1 else size[2] // 2 - 1

    X, Y, Z, _ = schematic.shape

    Mx = clamp(x + offset_x_h, 0, X - 1)
    mx = clamp(x - offset_x_l, 0, X - 1)
    My = clamp(y + offset_y_h, 0, Y - 1)
    my = clamp(y - offset_y_l, 0, Y - 1)
    Mz = clamp(z + offset_z_h, 0, Z - 1)
    mz = clamp(z - offset_z_l, 0, Z - 1)

    # coordinates inside the center cube
    cmx = (mx - x) + offset_x_l
    cmy = (my - y) + offset_y_l
    cmz = (mz - z) + offset_z_l
    cMx = (Mx - x) + offset_x_l
    cMy = (My - y) + offset_y_l
    cMz = (Mz - z) + offset_z_l

    ## only take block id [0] and ignore meta id [1]
    cube[cmx : cMx + 1, cmy : cMy + 1, cmz : cMz + 1, 0] = schematic[
        mx : Mx + 1, my : My + 1, mz : Mz + 1, 0
    ]
    cube = mbi.check_block_id()(cube)
    return cube


def get_batch_voxel_center_cubes(locs, schematic, size):
    """
    Given a list of locations, this function assembles a minibatch by cutting a cube of size
    around each location from schematic.
    """
    return np.array([get_voxel_center_cube(loc, schematic, size) for loc in locs])


def make_voxels_center(schematic, cube, offset):
    """
    Try to move non-air voxels to the center of the cube.
    After moving, we also fill in additional voxels from schematic.
    (As a result, these additional voxels might appear in another chopped cube.)
    """
    size = cube.shape[0]
    xs, ys, zs = np.nonzero(cube[:, :, :, 0] > 0)
    if len(xs) == 0:  # all air voxels, which is possible for a training schematic
        return cube, offset
    # add offsets to translate to the schematic coordinates
    mx = (min(xs) + max(xs) + 1) // 2
    my = (min(ys) + max(ys) + 1) // 2
    mz = (min(zs) + max(zs) + 1) // 2
    return (
        get_voxel_center_cube((mx, my, mz), cube, size),
        (mx - size // 2 + offset[0], my - size // 2 + offset[1], mz - size // 2 + offset[2]),
    )


def get_batch_cubes(schematic, size):
    """
    Cut schematic into cubes of [size, size, size].
    Return the chopped cubes and their offsets in schematic
    """
    X, Y, Z, _ = schematic.shape
    locs = []
    offsets = []
    for x in range((X + size - 1) // size):
        for y in range((Y + size - 1) // size):
            for z in range((Z + size - 1) // size):
                loc = (x * size + size // 2, y * size + size // 2, z * size + size // 2)
                locs.append(loc)
                offsets.append((x * size, y * size, z * size))
                ## overlap by half size
                loc = ((x + 1) * size, (y + 1) * size, (z + 1) * size)
                locs.append(loc)
                offsets.append((x * size + size // 2, y * size + size // 2, z * size + size // 2))

    cubes = get_batch_voxel_center_cubes(locs, schematic, size)
    # NOTE: this will change the label distribution (and the class weights)!
    #    for i in range(len(cubes)):
    #        cubes[i], offsets[i] = make_voxels_center(schematic, cubes[i], offsets[i])

    return cubes, offsets


def make_center(cube, size):
    """
    Put the cube in the center of sizexsizexsize
    """
    X, Y, Z = cube.shape[:3]
    ret = get_voxel_center_cube((X // 2, Y // 2, Z // 2), np.expand_dims(cube, axis=-1), size)
    return ret[:, :, :, 0]
