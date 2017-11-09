"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import numpy as np
from numpy import sin, cos, deg2rad

DIRECTIONS = {
    "AWAY": np.array([0, 0, 1]),
    "FRONT": np.array([0, 0, 1]),
    "BACK": np.array([0, 0, -1]),
    "LEFT": np.array([1, 0, 0]),
    "RIGHT": np.array([-1, 0, 0]),
    "DOWN": np.array([0, -1, 0]),
    "UP": np.array([0, 1, 0]),
}


def transform(coords, yaw, pitch, inverted=False):
    """Coordinate transforms with respect to yaw/pitch of the viewer
       coords should be relative to the viewer *before* pitch/yaw transform
       If we want to transform any of DIRECTIONS back, then it would be inverted=True
    """
    # our yaw and pitch are clockwise in the standard coordinate system
    theta = deg2rad(-yaw)
    gamma = deg2rad(-pitch)

    # standard 3d coordinate system as in:
    # http://planning.cs.uiuc.edu/node101.html#fig:yawpitchroll
    # http://planning.cs.uiuc.edu/node102.html
    #    ^ z
    #    |
    #    |-----> y
    #   /
    #  V x
    rtheta = np.array([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]])

    rgamma = np.array([[cos(gamma), 0, sin(gamma)], [0, 1, 0], [-sin(gamma), 0, cos(gamma)]])

    # Minecraft world:
    #         ^ y
    #         |  ^ z
    #         | /
    #  x <----/
    x, y, z = coords
    # a b c is the standard coordinate system
    if not inverted:
        trans_mat = np.linalg.inv(rtheta @ rgamma)
    else:
        trans_mat = rtheta @ rgamma
    a, b, c = trans_mat @ [-z, -x, y]
    # transform back to Minecraft world
    return [-b, c, -a]


def look_vec(yaw, pitch):
    yaw = deg2rad(yaw)
    pitch = deg2rad(pitch)
    x = -cos(pitch) * sin(yaw)
    y = -sin(pitch)
    z = cos(pitch) * cos(yaw)
    return np.array([x, y, z])


if __name__ == "__main__":
    A = (4, 0, 1)
    B = (4, 4, 4)
    print(transform(DIRECTIONS["RIGHT"], 45, 0, inverted=True))
