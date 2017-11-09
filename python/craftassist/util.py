from collections import defaultdict
from math import sin, cos, pi
import binascii
import hashlib
import logging
import numpy as np
import time
import traceback
from typing import cast, Tuple, List, TypeVar, Sequence

import rotation

XYZ = Tuple[int, int, int]
IDM = Tuple[int, int]
Block = Tuple[XYZ, IDM]
Hole = Tuple[List[XYZ], IDM]
T = TypeVar("T")  # generic type


def pos_to_np(pos):
    """Convert pos to numpy array"""
    if pos is None:
        return None
    return np.array((pos.x, pos.y, pos.z))


def to_block_pos(array):
    """Convert array to block position"""
    return np.floor(array).astype("int32")


def to_block_center(array):
    """Return the array centered at [0.5, 0.5, 0.5]"""
    return to_block_pos(array).astype("float") + [0.5, 0.5, 0.5]


def adjacent(p):
    """Return the positions adjacent to position p"""
    return (
        (p[0] + 1, p[1], p[2]),
        (p[0] - 1, p[1], p[2]),
        (p[0], p[1] + 1, p[2]),
        (p[0], p[1] - 1, p[2]),
        (p[0], p[1], p[2] + 1),
        (p[0], p[1], p[2] - 1),
    )


def diag_adjacent(p):
    """Return the adjacent positions to p including diagonal adjaceny"""
    return [
        (x, y, z)
        for x in range(p[0] - 1, p[0] + 2)
        for y in range(p[1] - 1, p[1] + 2)
        for z in range(p[2] - 1, p[2] + 2)
        if (x, y, z) != p
    ]


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


def shasum_file(path):
    """Retrn shasum of the file at path"""
    sha = hashlib.sha1()
    with open(path, "rb") as f:
        block = f.read(2 ** 16)
        while len(block) != 0:
            sha.update(block)
            block = f.read(2 ** 16)
    return binascii.hexlify(sha.digest())


def strip_idmeta(blockobj):
    """Return a list of (x, y, z) and drop the id_meta for blockobj"""
    if blockobj is not None:
        if type(blockobj) is dict:
            return list(pos for (pos, id_meta) in blockobj.items())
        else:
            return list(pos for (pos, id_meta) in blockobj)
    else:
        return None


def object_looked_at(
    agent, candidates: Sequence[Tuple[XYZ, T]], player, limit=1, max_distance=30
) -> List[Tuple[XYZ, T]]:
    """Return the object that `player` is looking at

    Args:
    - agent: agent object, for API access
    - candidates: list of (centroid, object) tuples
    - player: player struct whose POV to use for calculation
    - limit: 'ALL' or int; max candidates to return

    Returns: a list of (xyz, mem) tuples, max length `limit`
    """
    if len(candidates) == 0:
        return []

    pos = pos_to_np(player.pos)
    yaw, pitch = player.look.yaw, player.look.pitch

    # append to each candidate its relative position to player, rotated to
    # player-centric coordinates
    candidates_ = [(p, obj, rotation.transform(p - pos, yaw, pitch)) for (p, obj) in candidates]
    FRONT = rotation.DIRECTIONS["FRONT"]
    LEFT = rotation.DIRECTIONS["LEFT"]
    UP = rotation.DIRECTIONS["UP"]

    # reject objects behind player or not in cone of sight (but always include
    # an object if it's directly looked at)
    xsect = tuple(capped_line_of_sight(agent, player, 25))
    candidates_ = [
        (p, o, r)
        for (p, o, r) in candidates_
        if xsect in getattr(o, "blocks", {})
        or r @ FRONT > ((r @ LEFT) ** 2 + (r @ UP) ** 2) ** 0.5
    ]

    # if looking directly at an object, sort by proximity to look intersection
    if euclid_dist(pos, xsect) <= 25:
        candidates_.sort(key=lambda c: euclid_dist(c[0], xsect))
    else:
        # otherwise, sort by closest to look vector
        candidates_.sort(key=lambda c: ((c[2] @ LEFT) ** 2 + (c[2] @ UP) ** 2) ** 0.5)
    # linit returns of things too far away
    candidates_ = [c for c in candidates_ if euclid_dist(pos, c[0]) < max_distance]
    # limit number of returns
    if limit == "ALL":
        limit = len(candidates_)
    return [(p, o) for (p, o, r) in candidates_[:limit]]


def capped_line_of_sight(agent, player, cap=20):
    """Return the block directly in the entity's line of sight, or a point in the distance"""
    xsect = agent.get_player_line_of_sight(player)
    if xsect is not None and euclid_dist(pos_to_np(xsect), pos_to_np(player.pos)) <= cap:
        return pos_to_np(xsect)

    # default to cap blocks in front of entity
    vec = rotation.look_vec(player.look.yaw, player.look.pitch)
    return cap * np.array(vec) + to_block_pos(pos_to_np(player.pos))


def get_locs_from_entity(e):
    """Assumes input is either mob, memory, or tuple/list of coords
    outputs a tuple of coordinate tuples"""
    if str(type(e)).find("memory") > 0:
        if hasattr(e, "blocks"):
            return strip_idmeta(e.blocks)
        if hasattr(e, "position_history"):
            if len(e.position_history) > 0:
                mid = max(e.position_history)
                loc = e.position_history[mid]
                return tuple((tuple(to_block_pos(pos_to_np(loc))),))
        return None
    elif type(e) is tuple or type(e) is list:
        if len(e) > 0:
            if type(e[0]) is tuple:
                return e
            else:
                return tuple((e,))
        elif str(type(e)) == "<class 'agent.Mob'>" or str(type(e)).find("Agent") > 0:
            return tuple((tuple(to_block_pos(pos_to_np(e.pos))),))

        return None


def fill_idmeta(agent, poss: List[XYZ]) -> List[Block]:
    """Add id_meta information to a a list of (xyz)s"""
    if len(poss) == 0:
        return []
    mx, my, mz = np.min(poss, axis=0)
    Mx, My, Mz = np.max(poss, axis=0)
    B = agent.get_blocks(mx, Mx, my, My, mz, Mz)
    idms = []
    for x, y, z in poss:
        idm = tuple(B[y - my, z - mz, x - mx])
        idms.append(cast(IDM, idm))
    return [(cast(XYZ, tuple(pos)), idm) for (pos, idm) in zip(poss, idms)]


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
