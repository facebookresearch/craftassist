import numpy as np
import os
import sys
import random

VISION_DIR = os.path.dirname(os.path.realpath(__file__))
PYTHON_DIR = os.path.join(VISION_DIR, "../")
STACK_AGENT_DIR = os.path.join(PYTHON_DIR, "craftassist")
sys.path.append(STACK_AGENT_DIR)
import shapes
import random_shape_helpers as sh
import torch
import torchvision.utils as tvu

PERM = torch.randperm(256)
r = np.arange(0, 256) / 256
CMAP = np.stack((r, np.roll(r, 80), np.roll(r, 160)))


SMALL = 5
LARGE = 20


def options_cube():
    return {"size": np.random.randint(SMALL, LARGE), "orient": sh.orientation3()}


def options_hollow_cube():
    opts = {}
    opts["size"] = np.random.randint(SMALL, LARGE)
    opts["thickness"] = np.random.randint(1, opts["size"] - 3)
    return opts


def options_rectanguloid():
    return {"size": np.random.randint(SMALL, LARGE, size=3)}


def options_hollow_rectanguloid():
    opts = {}
    opts["size"] = np.random.randint(SMALL, LARGE, size=3)
    ms = min(opts["size"])
    opts["thickness"] = np.random.randint(1, ms - 3)
    return opts


def options_sphere():
    return {"radius": np.random.randint(SMALL, LARGE)}


def options_spherical_shell():
    opts = {}
    opts["radius"] = np.random.randint(SMALL, LARGE)
    opts["thickness"] = np.random.randint(1, opts["radius"] - 3)
    return opts


def options_square_pyramid():
    opts = {}
    opts["radius"] = np.random.randint(SMALL, LARGE)
    opts["slope"] = np.random.rand() * 0.4 + 0.8
    fullheight = opts["radius"] * opts["slope"]
    opts["height"] = np.random.randint(0.5 * fullheight, fullheight)
    return opts


def options_square():
    return {"size": np.random.randint(SMALL, LARGE), "orient": sh.orientation3()}


def options_rectangle():
    return {"size": np.random.randint(SMALL, LARGE, size=2), "orient": sh.orientation3()}


def options_circle():
    return {"radius": np.random.randint(SMALL, LARGE), "orient": sh.orientation3()}


def options_disk():
    return {"radius": np.random.randint(SMALL, LARGE), "orient": sh.orientation3()}


def options_triangle():
    return {"size": np.random.randint(SMALL, LARGE), "orient": sh.orientation3()}


def options_dome():
    return {"radius": np.random.randint(SMALL, LARGE)}


def options_arch():
    return {"size": np.random.randint(SMALL, LARGE), "distance": 2 * np.random.randint(2, 5) + 1}


def options_ellipsoid():
    return {"size": np.random.randint(SMALL, LARGE, size=3)}


def options_tower():
    return {"height": np.random.randint(3, 30), "base": np.random.randint(-4, 6)}


def options_empty():
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
SHAPENAMES.append("empty")
SHAPEFNS = sh.SHAPE_FNS
SHAPEFNS["TOWER"] = shapes.tower
SHAPEFNS["empty"] = empty

# 'flat',
# 'diag_flat',
# 'curved',
PARTNAMES = [
    "inside",
    "rect_top_corner",
    "rect_inner_top_corner",
    "rect_bottom_corner",
    "rect_inner_bottom_corner",
    "pyramid_peak",
    "pyramid_bottom_corner",
    "pyramid_bottom_edge",
    "pyramid_diag_edge",
    "rect_bottom_edge",
    "rect_inner_bottom_edge",
    "rect_vertical_edge",
    "rect_top_edge",
    "rect_inner_top_edge",
    "rect_inner_vertical_edge",
    "unk_part",
]

ALLLABELS = SHAPENAMES + PARTNAMES

LABELDICT = {v: k for k, v in dict(enumerate(ALLLABELS)).items()}

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


def draw_slices(c, L, vis):
    X = np.zeros((c.shape[0], 3, c.shape[1], c.shape[2]))
    Y = np.zeros((L[0].shape[0], 3, L[0].shape[1], L[0].shape[2]))
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            for k in range(c.shape[2]):
                for l in range(3):
                    X[j, l, i, k] = CMAP[l, PERM[int(c[i, j, k])]]
    idx = np.transpose(np.nonzero(L[0]))
    for i in idx:
        for l in range(3):
            Y[i[1], l, i[0], i[2]] += CMAP[l, PERM[int(i[3])]]
    img = tvu.make_grid(torch.from_numpy(X), nrow=8, normalize=True, scale_each=False)
    vis.image(img=img)
    img = tvu.make_grid(torch.from_numpy(Y), nrow=4, normalize=True, scale_each=False)
    vis.image(img=img)
    #    vis.images(Y, nrow=4)
    return X, Y


def get_shift(gridsize, schemasize, simple_center):
    if simple_center:
        return (gridsize - schemasize) // 2
    return np.random.randint(gridsize - schemasize)


# outputs a dense voxel rep from a sparse one.
# size should be a tuple of (H, W, D) for the desired voxel representation
# L should be a dict of lists with keys given by (x,y,z) as in blocks
#    each value of L is a list of tags associated with that location
# ldict maps the tags into integer class values
# shapename is a label for the whole object
# useid=True puts the block id into the voxel representation,
#    otherwise put a 1
def densify(blocks, size, L, ldict, shapename, useid=False, simple_center=True):
    bounds = shapes.get_bounds(blocks)
    dX = bounds[1] - bounds[0]
    dY = bounds[3] - bounds[2]
    dZ = bounds[5] - bounds[4]
    V = np.zeros((size[0], size[1], size[2]), dtype="int32")
    labels = []
    if L is not None:
        nlabels = len(ldict)
        for i in range(4):
            f = 2 ** (i + 1)
            labels.append(
                np.zeros((size[0] // f, size[1] // f, size[2] // f, nlabels), dtype="int32")
            )

    sx = get_shift(size[0], dX, simple_center)
    sy = get_shift(size[1], dY, simple_center)
    sz = get_shift(size[2], dZ, simple_center)

    offsets = (-bounds[0] + sx, -bounds[2] + sy, -bounds[4] + sz)
    for b in blocks:
        x = b[0][0] + offsets[0]
        y = b[0][1] + offsets[1]
        z = b[0][2] + offsets[2]
        if x >= 0 and y >= 0 and z >= 0 and x < size[0] and y < size[1] and z < size[2]:
            if type(b[1]) is int:
                V[x, y, z] = b[1]
            else:
                V[x, y, z] = b[1][0]
            if L is not None:
                for i in range(2, 4):
                    f = 2 ** (i + 1)
                    # TODO do this better.  e.g. peak could be at all scales
                    labels[i][x // f, y // f, z // f, ldict[shapename]] = 1
    if L is not None:
        for i in range(2):
            f = 2 ** (i + 1)
            for l in L:
                x = l[0] + offsets[0]
                y = l[1] + offsets[1]
                z = l[2] + offsets[2]
                if x >= 0 and y >= 0 and z >= 0:
                    if x < size[0] and y < size[1] and z < size[2]:
                        for ll in L[l]:
                            labels[i][x // f, y // f, z // f, ldict[ll]] = 1
    if not useid:
        V[V > 0] = 1

    return V, labels, offsets


def get_shape():
    name = random.choice(SHAPENAMES)
    opts = SHAPE_HELPERS[name]()
    opts["labelme"] = True
    return SHAPEFNS[name](**opts), opts, name


class ShapeDataset(torch.utils.data.Dataset):
    def __init__(self, nexamples=10000, sidelength=32, useid=False, mode="densify"):
        self.nexamples = nexamples
        self.sidelength = sidelength
        self.useid = useid
        self.mode = mode

    def __getitem__(self, index):
        sl = self.sidelength
        shape, shape_opts, shape_name = get_shape()
        shape_blocks = shape[0]
        block_labels = shape[1]
        if self.mode == "densify":
            ss, ll, _ = densify(
                shape_blocks, (sl, sl, sl), block_labels, LABELDICT, shape_name, useid=self.useid
            )
            return ss, ll
        elif self.mode == "bgraham":
            xyz = tuple(b[0] for b in shape_blocks)
            if self.useid:
                bids = tuple(b[1] for b in shape_blocks)
            else:
                bids = tuple(1 for b in shape_blocks)
            return xyz, bids, SHAPENAMES.index(shape_name)
        else:
            return shape_blocks, block_labels, SHAPENAMES.index(shape_name)

    def __len__(self):
        return self.nexamples


if __name__ == "__main__":
    import os
    import argparse
    import visdom

    vis = visdom.Visdom(server="http://localhost")
    sys.path.append(PYTHON_DIR)
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=32)
    opts = parser.parse_args()
    s = ShapeDataset(mode="bgraham")
    for i in range(500):
        u = s[0]
