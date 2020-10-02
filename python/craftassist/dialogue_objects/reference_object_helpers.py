"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import numpy as np
import rotation
import shapes

import heuristic_perception
from mc_util import pos_to_np, to_block_center, to_block_pos, ErrorWithResponse


def post_process_loc(loc, interpreter):
    return to_block_pos(loc)


def compute_locations(
    interpreter,
    speaker,
    mems,
    steps,
    reldir,
    repeat_num=1,
    repeat_dir=None,
    objects=[],
    padding=(1, 1, 1),
    enable_geoscorer=False,
):
    agent = interpreter.agent
    repeat_num = max(repeat_num, len(objects))
    player_look = agent.perception_modules["low_level"].get_player_struct_by_name(speaker).look
    player_pos = pos_to_np(agent.get_player().pos)
    origin = compute_location_heuristic(player_look, player_pos, mems, steps, reldir)
    if (
        enable_geoscorer
        and agent.geoscorer is not None
        and agent.geoscorer.use(steps, repeat_num, reldir)
    ):
        r = agent.geoscorer.radius
        brc = (origin[0] - r, origin[1] - r, origin[2] - r)
        tlc = (brc[0] + 2 * r - 1, brc[1] + 2 * r - 1, brc[2] + 2 * r - 1)
        context = agent.get_blocks(brc[0], tlc[0], brc[1], tlc[1], brc[2], tlc[2])
        segment = objects[0][0]
        origin = agent.geoscorer.produce_segment_pos_in_context(segment, context, brc)
        offsets = [(0, 0, 0)]
    else:
        if repeat_num > 1:
            schematic = None if len(objects) == 0 else objects[0][0]
            offsets = get_repeat_arrangement(
                player_look, repeat_num, repeat_dir, mems, schematic, padding
            )
        else:
            offsets = [(0, 0, 0)]
    origin = post_process_loc(origin, interpreter)
    offsets = [post_process_loc(o, interpreter) for o in offsets]
    return origin, offsets


# There will be at least one mem in mems
def compute_location_heuristic(player_look, player_pos, mems, steps, reldir):
    loc = mems[0].get_pos()
    if reldir is not None:
        steps = steps or 5
        if reldir == "BETWEEN":
            loc = (np.add(mems[0].get_pos(), mems[1].get_pos())) / 2
            loc = (loc[0], loc[1], loc[2])
        elif reldir == "INSIDE":
            for i in range(len(mems)):
                mem = mems[i]
                locs = heuristic_perception.find_inside(mem)
                if len(locs) > 0:
                    break
            if len(locs) == 0:
                raise ErrorWithResponse("I don't know how to go inside there")
            else:
                loc = locs[0]
        elif reldir == "AWAY":
            dir_vec = (player_pos - loc) / np.linalg.norm(player_pos - loc)
            loc = steps * np.array(dir_vec) + to_block_center(loc)
        elif reldir == "NEAR":
            pass
        elif reldir == "AROUND":
            pass
        else:  # LEFT, RIGHT, etc...
            reldir_vec = rotation.DIRECTIONS[reldir]
            # this should be an inverse transform so we set inverted=True
            dir_vec = rotation.transform(reldir_vec, player_look.yaw, 0, inverted=True)
            loc = steps * np.array(dir_vec) + to_block_center(loc)
    elif steps is not None:
        loc = to_block_center(loc) + [0, 0, steps]
    return to_block_pos(loc)


def get_repeat_arrangement(
    player_look, repeat_num, repeat_dir, ref_mems, schematic=None, padding=(1, 1, 1)
):
    shapeparams = {}
    # default repeat dir is LEFT
    if not repeat_dir:
        repeat_dir = "LEFT"
    # eventually fix this to allow number based on shape
    shapeparams["N"] = repeat_num

    if repeat_dir == "AROUND":
        # TODO vertical "around"
        shapeparams["orient"] = "xy"
        shapeparams["extra_space"] = max(padding)
        central_object = ref_mems[0]
        bounds = central_object.get_bounds()
        b = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
        shapeparams["encircled_object_radius"] = b

        offsets = shapes.arrange("circle", schematic, shapeparams)
    else:

        reldir_vec = rotation.DIRECTIONS[repeat_dir]
        # this should be an inverse transform so we set inverted=True
        dir_vec = rotation.transform(reldir_vec, player_look.yaw, 0, inverted=True)
        max_ind = np.argmax(dir_vec)
        shapeparams["extra_space"] = padding[max_ind]
        shapeparams["orient"] = dir_vec
        offsets = shapes.arrange("line", schematic, shapeparams)
    offsets = [tuple(to_block_pos(o)) for o in offsets]
    return offsets
