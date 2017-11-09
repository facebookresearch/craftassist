import numpy as np
import sys

sys.path.append("..")
sys.path.append("../..")
from voxel_cube import get_voxel_center_cube


class Anchor(object):
    def __init__(self, x, y, z, X, Y, Z):
        self.x = x
        self.y = y
        self.z = z
        self.X = X
        self.Y = Y
        self.Z = Z

    def __repr__(self):
        return "{} {} {} {} {} {}".format(self.x, self.y, self.z, self.X, self.Y, self.Z)

    @property
    def center(self):
        cx = self.x + self.X // 2
        cy = self.y + self.Y // 2
        cz = self.z + self.Z // 2
        return (cx, cy, cz)

    @property
    def tx(self):
        return self.x + self.X - 1

    @property
    def ty(self):
        return self.y + self.Y - 1

    @property
    def tz(self):
        return self.z + self.Z - 1

    def volume(self, axis=None):
        if axis == 0:
            return self.Y * self.Z
        elif axis == 1:
            return self.X * self.Z
        elif axis == 2:
            return self.X * self.Y
        else:
            return self.X * self.Y * self.Z

    def intersection(self, anchor):
        mx = max(self.x, anchor.x)
        my = max(self.y, anchor.y)
        mz = max(self.z, anchor.z)
        Mx = min(self.tx, anchor.tx)
        My = min(self.ty, anchor.ty)
        Mz = min(self.tz, anchor.tz)
        X = Mx - mx + 1
        Y = My - my + 1
        Z = Mz - mz + 1
        if X > 0 and Y > 0 and Z > 0:
            return Anchor(mx, my, mz, X, Y, Z)


def generate_anchors(N, S, sizes_x, ratios_y, ratios_z):
    """
    Return a list of anchors given a cube of size NxNxN and a stride S.

    Input:
        N: the cube size
        S: the stride between two neighboring anchors
        size_x: a list of possible sizes for x-axis
        ratios_y: a list of aspect ratios for y-axis
        ratios_z: a list of aspect ratios for z-axis

    Output:
        anchors: a list of Anchor objects
    """
    assert N % S == 0
    M = N // S

    def generate_anchor(i, j, k, X, ry, rz):
        cx = i + 0.5
        cy = j + 0.5
        cz = k + 0.5
        Y = ry * X
        Z = rz * X
        x = cx - X / 2
        y = cy - Y / 2
        z = cz - Z / 2
        return Anchor(int(S * x), int(S * y), int(S * z), int(S * X), int(S * Y), int(S * Z))

    anchors = [
        generate_anchor(i, j, k, X, ry, rz)
        for i in range(M)
        for j in range(M)
        for k in range(M)
        for X in sizes_x
        for ry in ratios_y
        for rz in ratios_z
    ]
    return anchors


def generate_anchor_groundtruth(N, S, sizes_x, ratios_y, ratios_z, house_annotation, threshold):
    """
    Given a house segment annotation, generate a set of anchors (ROI candidates) with
    their labels being either 0 or 1. This function will check each anchor against the
    annotated segments and use the threshold to decide whether it is positive or
    negative.

    Input:
        N: the cube has a size of NxNxN
        S: anchor stride
        house_annotation: a 3D numpy array, where each non-positive id represents
                          an annotated segment (not necessarily continuous).

    Output:
        annotated_anchors: a list of tuples, where each tuple is a triplet (Anchor, 0/1, mask).
                 0 represents negative and 1 represents positive. The order of tuples should
                 be consistent with the anchor order on the (N/S)x(N/S)x(N/S)xK feature map.
    """

    def major_id(cube):
        cube = cube[np.nonzero(cube > 0)]
        if len(cube) == 0:
            return 0
        return np.argmax(np.bincount(cube))

    def test_overlap(mask, anchor):
        xs, ys, zs = np.nonzero(mask)
        mx, Mx = min(xs), max(xs)
        my, My = min(ys), max(ys)
        mz, Mz = min(zs), max(zs)
        a = Anchor(mx, my, mz, Mx - mx + 1, My - my + 1, Mz - mz + 1)
        intersection = a.intersection(anchor)

        def overlap(axis):
            return intersection.volume(axis) / (
                anchor.volume(axis) + a.volume(axis) - intersection.volume(axis)
            )

        # if overlap(0) >= threshold \
        #    or overlap(1) >= threshold \
        #    or overlap(2) >= threshold:
        if overlap(axis=None) >= threshold:
            mask = get_voxel_center_cube(
                anchor.center, np.expand_dims(mask, axis=-1), (anchor.X, anchor.Y, anchor.Z)
            )
            return np.squeeze(mask, axis=-1)

    anchors = generate_anchors(N, S, sizes_x, ratios_y, ratios_z)
    annotated_anchors = []
    for a in anchors:
        l = a.center
        #        inst_id = major_id(house_annotation[a.x:a.tx+1, a.y:a.ty+1, a.z:a.tz+1])
        inst_id = house_annotation[l[0], l[1], l[2]]
        mask = None
        if inst_id > 0:
            mask = test_overlap((house_annotation == inst_id).astype(np.float32), a)
        annotated_anchors.append([a, mask is not None, mask])
    return annotated_anchors


def anchorify_parallel(args):
    anno, N, S, sizes_x, ratios_y, ratios_z, threshold = args
    annotated_anchors = generate_anchor_groundtruth(
        N, S, sizes_x, ratios_y, ratios_z, anno, threshold
    )
    return annotated_anchors
