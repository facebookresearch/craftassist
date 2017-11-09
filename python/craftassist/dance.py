"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

"""This file has functions to implement different dances for the agent.
"""
import time
import numpy as np
import tasks
import shapes
import search
from util import ErrorWithResponse


konami_dance = [
    (0, 1, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, -1, 0),
    (0, 0, -1),
    (0, 0, 1),
    (0, 0, -1),
    (0, 0, 1),
]


def add_default_dances(memory):
    memory.add_dance(generate_sequential_move_fn(konami_dance))


def generate_sequential_move_fn(sequence):
    def move_fn(danceObj, agent, dance_location):
        if danceObj.tick >= len(sequence):
            return None
        else:
            dpos = sequence[danceObj.tick]
            danceObj.tick += 1
            target_location = dance_location if dance_location is not None else agent.pos
            target = target_location + dpos
            mv = tasks.Move(agent, {"target": target, "approx": 0})
        return mv

    return move_fn


class Movement(object):
    def __init__(self, agent, move_fn, dance_location=None):
        self.agent = agent
        self.move_fn = move_fn
        self.dance_location = dance_location
        self.tick = 0
        self.time = time.time()

    def get_move(self):
        # move_fn should output a tuple (dx, dy, dz) corresponding to a
        # change in Movement or None
        # if None then Movement is finished
        # can output
        return self.move_fn(self, self.agent, self.dance_location)


class SequentialMove(Movement):
    def __init__(self, agent, sequence):
        move_fn = generate_sequential_move_fn(sequence)
        super(SequentialMove, self).__init__(agent, move_fn)


# TODO: class TimedDance(Movement):


# e.g. go around the x
#     go through the x
#     go over the x
#     go across the x
class RefObjMovement(Movement):
    def __init__(
        self,
        agent,
        ref_object=None,
        relative_direction="CLOCKWISE",  # this is the memory of the object
    ):
        self.agent = agent
        self.tick = 0
        blocks = [(bpos, bid) for bpos, bid in ref_object.blocks.items()]
        bounds = shapes.get_bounds(blocks)
        center = np.mean([b[0] for b in blocks], axis=0)

        d = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
        if relative_direction == "CLOCKWISE":
            offsets = shapes.arrange(
                "circle", schematic=None, shapeparams={"encircled_object_radius": d}
            )
        elif relative_direction == "ANTICLOCKWISE":
            offsets = shapes.arrange(
                "circle", schematic=None, shapeparams={"encircled_object_radius": d}
            )
            offsets = offsets[::-1]
        else:
            raise NotImplementedError("TODO other kinds of paths")
        self.path = [np.round(center + o) for o in offsets]
        self.path.append(self.path[0])

        # check each offset to find a nearby reachable point, see if a path
        # is possible now, and error otherwise

        for i in range(len(self.path) - 1):
            path = search.astar(agent, self.path[i + 1], approx=2, pos=self.path[i])
            if path is None:
                raise ErrorWithResponse("I cannot find an appropriate path.")

    def get_move(self):
        if self.tick >= len(self.path):
            return None
        mv = tasks.Move(self.agent, {"target": self.path[self.tick], "approx": 2})
        self.tick += 1
        return mv
