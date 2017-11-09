import numpy as np
import random


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


# outputs a dense voxel rep from a sparse one.
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
    return abs(b[1] - a[1]) <= d[1] and abs(b[1] - a[1]) <= d[1] and abs(b[1] - a[1]) <= d[2]


############################################################################
# For these "S" is a list of blocks in ((x,y,z),(id, meta)) format
# the segment is a list of the same length as S with either True or False
# at each entry marking whether that block is in the segment
# each outputs a list of blocks in ((x,y,z),(id, meta)) format


def shift_negative(S, segment, args):
    shift_max = args["shift_max"]
    """takes the blocks not in the sgement and shifts them randomly"""
    shift_vec = random_int_triple(-shift_max, shift_max)
    N = []
    for s in range(len(segment)):
        if not segment[s]:
            new_coords = tuple(np.add(S[s][0], shift_vec))
            N.append([new_coords, S[s][1]])
        else:
            N.append(S[s])
    return N


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
