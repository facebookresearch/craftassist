"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import logging
import numpy as np
from typing import Dict, List

from util import XYZ, IDM, Block
from utils import Player, Look, Pos, Item
from craftassist_agent import CraftAssistAgent


class FakeAgent:
    CCW_LOOK_VECS = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def __init__(self, memory):
        self.memory = memory
        self.no_default_behavior = True
        self.last_task_memid = None
        self.pos = np.array((0, 63, 0), dtype="int")

        self._world: Dict[XYZ, IDM] = {}
        self._held_item: IDM = (0, 0)
        self._look_vec = (1, 0)  # (x, z) unit vec
        self._changed_blocks: List[Block] = []
        self._outgoing_chats: List[str] = []

    ####################
    ##  TEST METHODS  ##
    ####################

    def clear_outgoing_chats(self):
        self._outgoing_chats.clear()

    def get_last_outgoing_chat(self):
        try:
            return self._outgoing_chats[-1]
        except IndexError:
            return None

    ########################
    ##  FAKE .PY METHODS  ##
    ########################

    def task_step(self):
        CraftAssistAgent.task_step(self, sleep_time=0)

    def point_at(*args):
        pass

    ########################
    ##  FAKE C++ METHODS  ##
    ########################

    def dig(self, x, y, z):
        try:
            del self._world[(x, y, z)]
            self._changed_blocks.append(((x, y, z), (0, 0)))
            return True
        except KeyError:
            return False

    def send_chat(self, chat):
        logging.info("FakeAgent.send_chat: {}".format(chat))
        self._outgoing_chats.append(chat)

    def set_held_item(self, arg):
        try:
            d, m = arg
            self._held_item = (d, m)
        except TypeError:
            self._held_item = (arg, 0)

    def step_pos_x(self):
        self.pos += (1, 0, 0)

    def step_neg_x(self):
        self.pos += (-1, 0, 0)

    def step_pos_z(self):
        self.pos += (0, 0, 1)

    def step_neg_z(self):
        self.pos += (0, 0, -1)

    def step_pos_y(self):
        self.pos += (0, 1, 0)

    def step_neg_y(self):
        self.pos += (0, -1, 0)

    def step_forward(self):
        dx, dz = self._look_vec
        self.pos += (dx, 0, dz)

    def look_at(self, x, y, z):
        raise NotImplementedError()

    def set_look(self, look):
        raise NotImplementedError()

    def turn_angle(self, angle):
        if angle == 90:
            self.turn_left()
        elif angle == -90:
            self.turn_right()
        else:
            raise ValueError("bad angle={}".format(angle))

    def turn_left(self):
        idx = self.CCW_LOOK_VECS.index(self._look_vec)
        self._look_vec = self.CCW_LOOK_VECS[(idx + 1) % len(self.CCW_LOOK_VECS)]

    def turn_right(self):
        idx = self.CCW_LOOK_VECS.index(self._look_vec)
        self._look_vec = self.CCW_LOOK_VECS[(idx - 1) % len(self.CCW_LOOK_VECS)]

    def place_block(self, x, y, z):
        self._world[(x, y, z)] = self._held_item
        self._changed_blocks.append(((x, y, z), self._held_item))
        return True

    def craft(self):
        raise NotImplementedError()

    def get_blocks(self, xa, xb, ya, yb, za, zb):
        xw = xb - xa + 1
        yw = yb - ya + 1
        zw = zb - za + 1
        B = np.zeros((yw, zw, xw, 2), dtype="uint8")
        for x in range(xa, xb + 1):
            for y in range(ya, yb + 1):
                for z in range(za, zb + 1):
                    b = self._world.get((x, y, z), (0, 0) if y >= 63 else (2, 0))
                    B[y - ya, z - za, x - xa, :] = b
        return B

    def get_local_blocks(self, r):
        x, y, z = self.pos
        return self.get_blocks(x - r, x + r, y - r, y + r, z - r, z + r)

    def get_incoming_chats(self):
        raise NotImplementedError()

    def get_player(self):
        return Player(1, "fake_agent", Pos(*self.pos), Look(0, 0), Item(0, 0))

    def get_mobs(self):
        return []

    def get_other_players(self):
        return [Player(42, "SPEAKER", Pos(5, 63, 5), Look(270, 0), Item(0, 0))]

    def get_other_player_by_name(self):
        raise NotImplementedError()

    def get_vision(self):
        raise NotImplementedError()

    def get_line_of_sight(self):
        raise NotImplementedError()

    def get_player_line_of_sight(self):
        raise NotImplementedError()

    def get_changed_blocks(self) -> List[Block]:
        r = self._changed_blocks.copy()
        self._changed_blocks.clear()
        return r
