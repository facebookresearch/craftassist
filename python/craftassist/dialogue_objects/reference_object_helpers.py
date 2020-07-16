"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import numpy as np
import rotation
import shapes

import heuristic_perception
from base_agent.dialogue_objects import SPEAKERLOOK, is_loc_speakerlook
from base_agent.util import ErrorWithResponse
from util import pos_to_np, to_block_center, to_block_pos
from word2number.w2n import word_to_num


def post_process_loc(loc, interpreter):
    return to_block_pos(loc)


def compute_locations(interpreter, speaker, d, ref_mems, objects=[], enable_geoscorer=False):
    location_d = d.get("location", SPEAKERLOOK)
    repeat_num = len(objects)
    origin = compute_location_heuristic(interpreter, speaker, location_d, ref_mems)
    if (
        enable_geoscorer
        and interpreter.agent.geoscorer is not None
        and interpreter.agent.geoscorer.use(location_d, repeat_num)
    ):
        r = interpreter.agent.geoscorer.radius
        brc = (origin[0] - r, origin[1] - r, origin[2] - r)
        tlc = (brc[0] + 2 * r - 1, brc[1] + 2 * r - 1, brc[2] + 2 * r - 1)
        context = interpreter.agent.get_blocks(brc[0], tlc[0], brc[1], tlc[1], brc[2], tlc[2])
        segment = objects[0][0]
        origin = interpreter.agent.geoscorer.produce_segment_pos_in_context(segment, context, brc)
        origin = to_block_pos(origin)
        offsets = [(0, 0, 0)]
    else:
        # hack to fix build 1 block underground!!!
        # FIXME should SPEAKER_LOOK deal with this?
        if is_loc_speakerlook(location_d):
            origin[1] += 1

        if repeat_num > 1:
            offsets = get_repeat_arrangement(
                d, interpreter, speaker, objects[0][0], ref_mems, repeat_num=repeat_num
            )
        else:
            offsets = [(0, 0, 0)]
    return origin, offsets


def compute_location_heuristic(interpreter, speaker, d, mems):
    # handle relative direction
    reldir = d.get("relative_direction")
    loc = mems[0].get_pos()
    if reldir is not None:
        if reldir == "BETWEEN":
            loc = (np.add(mems[0].get_pos(), mems[1].get_pos())) / 2
            loc = (loc[0], loc[1], loc[2])
        elif reldir == "INSIDE":
            ref_obj_dict = d.get("reference_object", SPEAKERLOOK["reference_object"])
            special = ref_obj_dict.get("special_reference")
            if not special:
                for i in range(len(mems)):
                    mem = mems[i]
                    locs = heuristic_perception.find_inside(mem)
                    if len(locs) > 0:
                        break
                if len(locs) == 0:
                    raise ErrorWithResponse("I don't know how to go inside there")
                else:
                    interpreter.memory.update_recent_entities([mem])
                    loc = locs[0]
            else:
                raise ErrorWithResponse("I don't know how to go inside there")
        elif reldir == "AWAY":
            apos = pos_to_np(interpreter.agent.get_player().pos)
            dir_vec = (apos - loc) / np.linalg.norm(apos - loc)
            num_steps = word_to_num(d.get("steps", "5"))
            loc = num_steps * np.array(dir_vec) + to_block_center(loc)
        elif reldir == "NEAR":
            pass
        else:  # LEFT, RIGHT, etc...
            reldir_vec = rotation.DIRECTIONS[reldir]
            look = (
                interpreter.agent.perception_modules["low_level"]
                .get_player_struct_by_name(speaker)
                .look
            )
            # this should be an inverse transform so we set inverted=True
            dir_vec = rotation.transform(reldir_vec, look.yaw, 0, inverted=True)
            num_steps = word_to_num(d.get("steps", "5"))
            loc = num_steps * np.array(dir_vec) + to_block_center(loc)

    # if steps without relative direction
    elif "steps" in d:
        num_steps = word_to_num(d.get("steps", "5"))
        loc = to_block_center(loc) + [0, 0, num_steps]
    return post_process_loc(loc, interpreter)


def get_repeat_arrangement(
    d, interpreter, speaker, schematic, ref_mems, repeat_num=-1, extra_space=1
):
    shapeparams = {}
    # eventually fix this to allow number based on shape
    shapeparams["N"] = repeat_num
    shapeparams["extra_space"] = extra_space
    if "repeat" in d:
        direction_name = d.get("repeat", {}).get("repeat_dir", "FRONT")
    elif "schematic" in d:
        direction_name = d["schematic"].get("repeat", {}).get("repeat_dir", "FRONT")
    if direction_name != "AROUND":
        reldir_vec = rotation.DIRECTIONS[direction_name]
        look = (
            interpreter.agent.perception_modules["low_level"]
            .get_player_struct_by_name(speaker)
            .look
        )
        # this should be an inverse transform so we set inverted=True
        dir_vec = rotation.transform(reldir_vec, look.yaw, 0, inverted=True)
        shapeparams["orient"] = dir_vec
        offsets = shapes.arrange("line", schematic, shapeparams)
    else:
        # TODO vertical "around"
        shapeparams["orient"] = "xy"
        shapeparams["encircled_object_radius"] = 1
        if d.get("location") is not None:
            central_object = ref_mems[0]
            # FIXME: .blocks is unsafe, assumes BlockObject only object could be Mob. Ignoring for now.
            central_object_blocks = central_object[0].blocks  # type: ignore
            # .blocks returns a dict of (x, y, z) : (block_id, meta), convert to list
            # to get bounds
            central_object_list = [tuple([k, v]) for k, v in central_object_blocks.items()]
            bounds = shapes.get_bounds(central_object_list)
            b = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
            shapeparams["encircled_object_radius"] = b
        offsets = shapes.arrange("circle", schematic, shapeparams)
    offsets = [tuple(to_block_pos(o)) for o in offsets]
    return offsets
