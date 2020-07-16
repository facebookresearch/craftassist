"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

from collections import defaultdict, namedtuple
from math import sin, cos, pi
import binascii
import hashlib
import logging
import numpy as np
import time
import traceback
from typing import Tuple, List, TypeVar
import uuid

##FFS FIXME!!!! arrange utils properly, put things in one place
XYZ = Tuple[int, int, int]
# two points p0(x0, y0, z0), p1(x1, y1, z1) determine a 3d cube(point_at_target)
POINT_AT_TARGET = Tuple[int, int, int, int, int, int]
IDM = Tuple[int, int]
Block = Tuple[XYZ, IDM]
Hole = Tuple[List[XYZ], IDM]
T = TypeVar("T")  # generic type
#####FIXME!!!!!!  make all these dicts all through code
Pos = namedtuple("pos", ["x", "y", "z"])
Look = namedtuple("look", "yaw, pitch")
Player = namedtuple("Player", "entityId, name, pos, look")

TICK_PER_SEC = 100

# TODO make this just a dict, and change in memory and agent
# eg in object_looked_at and PlayerNode
def to_player_struct(pos, yaw, pitch, eid, name):
    if len(pos) == 2:
        pos = Pos(pos[0], 0.0, pos[1])
    else:
        pos = Pos(pos[0], pos[1], pos[2])
    look = Look(yaw, pitch)
    return Player(eid, name, pos, look)


# converts from seconds to internal tick
def round_time(t):
    return int(TICK_PER_SEC * t)


class Time:
    def __init__(self, mode="clock"):
        # mode is "clock" or "tick".  If "clock", converts seconds to ticks by rounding
        # if tick, returns number of ticks since start
        assert mode == "tick" or mode == "clock"
        self.mode = mode
        self.init_time_raw = time.time()
        self.time = 0

    def get_time(self):
        return self.round_time(self.time)

    def round_time(self, t):
        if self.mode == "tick":
            return t
        else:
            return round_time(time.time() - self.init_time_raw)

    def add_tick(self, ticks=1):
        if self.mode == "tick":
            self.time += 1
        else:
            time.sleep(ticks / TICK_PER_SEC)


def get_bounds(locs):
    M = np.max(locs, axis=0)
    m = np.min(locs, axis=0)
    return m[0], M[0], m[1], M[1], m[2], M[2]


def pos_to_np(pos):
    """Convert pos to numpy array"""
    if pos is None:
        return None
    return np.array((pos.x, pos.y, pos.z))


def discrete_step_dir(agent):
    """Discretized unit vector in the direction of agent's yaw

    agent pos + discrete_step_dir = block in front of agent
    """
    yaw = agent.get_player().look.yaw
    x = round(-sin(yaw * pi / 180))
    z = round(cos(yaw * pi / 180))
    return np.array([x, 0, z], dtype="int32")


def euclid_dist(a, b):
    """Return euclidean distance between a and b"""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def manhat_dist(a, b):
    """Return mahattan ditance between a and b"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def group_by(items, key_fn):
    """Return a dict of {k: list[x]}, where key_fn(x) == k"""
    d = defaultdict(list)
    for x in items:
        d[key_fn(x)].append(x)
    return d


def hash_user(username):
    """Encrypt username"""
    # uuid is used to generate a random number
    salt = uuid.uuid4().hex
    return hashlib.sha256(salt.encode() + username.encode()).hexdigest() + ":" + salt


def check_username(hashed_username, username):
    """Compare the username with the hash to check if they
    are same"""
    user, salt = hashed_username.split(":")
    return user == hashlib.sha256(salt.encode() + username.encode()).hexdigest()


def shasum_file(path):
    """Retrn shasum of the file at path"""
    sha = hashlib.sha1()
    with open(path, "rb") as f:
        block = f.read(2 ** 16)
        while len(block) != 0:
            sha.update(block)
            block = f.read(2 ** 16)
    return binascii.hexlify(sha.digest())


class ErrorWithResponse(Exception):
    def __init__(self, chat):
        self.chat = chat


class NextDialogueStep(Exception):
    pass


class TimingWarn(object):
    """Context manager which logs a warning if elapsed time exceeds some threshold"""

    def __init__(self, max_time: float):
        self.max_time = max_time

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.elapsed_time = time.time() - self.start_time
        if self.elapsed_time >= self.max_time:
            logging.warn(
                "Timing exceeded threshold: {}".format(self.elapsed_time)
                + "\n"
                + "".join(traceback.format_stack(limit=2))
            )
