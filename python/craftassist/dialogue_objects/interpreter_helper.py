"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import logging
import Levenshtein
import numpy as np
import random
import re
from typing import cast, List, Tuple, Union, Optional, Dict, Any

from base_agent.dialogue_objects import ConfirmReferenceObject
import block_data
import minecraft_specs
import heuristic_perception
import rotation
import shapes
import size_words
from mc_memory_nodes import MobNode
from base_agent.memory_nodes import PlayerNode, ReferenceObjectNode
from base_agent.stop_condition import StopCondition, NeverStopCondition
from mc_stop_condition import AgentAdjacentStopCondition

# FIXME!
from base_agent.util import euclid_dist, ErrorWithResponse, NextDialogueStep

from util import (
    Block,
    Hole,
    IDM,
    T,
    XYZ,
    most_common_idm,
    capped_line_of_sight,
    object_looked_at,
    pos_to_np,
    strip_idmeta,
    to_block_center,
    to_block_pos,
)
from word2number.w2n import word_to_num
from word_maps import SPECIAL_SHAPE_FNS, SPECIAL_SHAPES_CANONICALIZE


def tags_from_dict(d):
    return [
        strip_prefix(tag, "the ")
        for key, tag in d.items()
        if key.startswith("has_") and isinstance(tag, str)
    ]


def interpret_reference_object(
    interpreter, speaker, d, ignore_mobs=False, limit=1, loose_speakerlook=False
) -> List[ReferenceObjectNode]:
    if d.get("contains_coreference", "NULL") != "NULL":
        mem = d["contains_coreference"]
        if isinstance(mem, ReferenceObjectNode):
            return [mem]
        else:
            logging.error("bad coref_resolve -> {}".format(mem))

    if len(interpreter.progeny_data) == 0:
        tags = tags_from_dict(d)
        # TODO Add ignore_player maybe?
        candidates = (
            get_reference_objects(interpreter, *tags)
            if not ignore_mobs
            else get_objects(interpreter, *tags)
        )
        if len(candidates) > 0:
            location_d = d.get("location", {"location_type": "SPEAKER_LOOK"})
            if limit == 1:
                # override with input value
                limit = get_repeat_num(d)
            r = filter_by_sublocation(
                interpreter, speaker, candidates, location_d, limit=limit, loose=loose_speakerlook
            )
            return [mem for _, mem in r]
        else:
            # no candidates found; ask Clarification
            # TODO: move ttad call to dialogue manager and remove this logic
            interpreter.action_dict_frozen = True
            player_struct = interpreter.agent.perception_modules[
                "low_level"
            ].get_player_struct_by_name(speaker)
            confirm_candidates = get_objects(interpreter)  # no tags
            objects = object_looked_at(
                interpreter.agent, confirm_candidates, player_struct, limit=1
            )
            if len(objects) == 0:
                raise ErrorWithResponse("I don't know what you're referring to")
            _, mem = objects[0]
            blocks = list(mem.blocks.keys())
            interpreter.provisional["object_mem"] = mem
            interpreter.provisional["object"] = blocks
            interpreter.provisional["d"] = d
            interpreter.dialogue_stack.append_new(ConfirmReferenceObject, blocks)
            raise NextDialogueStep()

    else:
        # clarification answered
        r = interpreter.progeny_data[-1].get("response")
        if r == "yes":
            # TODO: learn from the tag!  put it in memory!
            return [interpreter.provisional.get("object_mem")] * limit
        else:
            # TODO: error handling here ?
            return []


def interpret_shape_schematic(
    interpreter, speaker, d, shapename=None
) -> Tuple[List[Block], List[Tuple[str, str]]]:
    """Return a tuple of 2 values:
    - the schematic blocks, list[(xyz, idm)]
    - a list of (pred, val) tags
    """
    if shapename is not None:
        shape = shapename
    else:
        # For sentences like "Stack" and "Place" that have the shapename in dict
        shape = d["has_shape"]

    numeric_keys = [
        "has_thickness",
        "has_radius",
        "has_depth",
        "has_width",
        "has_height",
        "has_length",
        "has_slope",
        #        "has_orientation", #is this supposed to be numeric key?
        "has_distance",
        "has_base",
    ]

    attrs = {key[4:]: word_to_num(d[key]) for key in numeric_keys if key in d}

    if "has_orientation" in d:
        attrs["orient"] = d["has_orientation"]

    if "has_size" in d:
        attrs["size"] = interpret_size(interpreter, d["has_size"])

    if "has_block_type" in d:
        block_type = get_block_type(d["has_block_type"])
        attrs["bid"] = block_type
    elif "has_colour" in d:
        c = block_data.COLOR_BID_MAP.get(d["has_colour"])
        if c is not None:
            attrs["bid"] = random.choice(c)

    tags = []
    for key, val in d.items():
        if key.startswith("has_"):
            stemmed_val = val
            tags.append((key, stemmed_val))

    return SPECIAL_SHAPE_FNS[shape](**attrs), tags


def interpret_size(interpreter, text) -> Union[int, List[int]]:
    """Processes the has_size_ span value and returns int or list[int]"""
    nums = re.findall("[-0-9]+", text)
    if len(nums) == 1:
        # handle "3", "three", etc.
        return word_to_num(nums[0])
    elif len(nums) > 1:
        # handle "3 x 3", "four by five", etc.
        return [word_to_num(n) for n in nums]
    else:
        # handle "big", "really huge", etc.
        if hasattr(interpreter.agent, "size_str_to_int"):
            return interpreter.agent.size_str_to_int(text)
        else:
            return size_words.size_str_to_int(text)


def interpret_named_schematic(
    interpreter, speaker, d
) -> Tuple[List[Block], Optional[str], List[Tuple[str, str]]]:
    """Return a tuple of 3 values:
    - the schematic blocks, list[(xyz, idm)]
    - a SchematicNode memid, or None
    - a list of (pred, val) tags
    """
    if "has_name" not in d:
        raise ErrorWithResponse("I don't know what you want me to build.")
    name = d["has_name"]
    stemmed_name = name
    shapename = SPECIAL_SHAPES_CANONICALIZE.get(name) or SPECIAL_SHAPES_CANONICALIZE.get(
        stemmed_name
    )
    if shapename:
        shape_blocks, tags = interpret_shape_schematic(
            interpreter, speaker, d, shapename=shapename
        )
        return shape_blocks, None, tags

    schematic = interpreter.memory.get_schematic_by_name(name)
    if schematic is None:
        schematic = interpreter.memory.get_schematic_by_name(stemmed_name)
        if schematic is None:
            raise ErrorWithResponse("I don't know what you want me to build.")
    tags = [(p, v) for (_, p, v) in interpreter.memory.get_triples(subj=schematic.memid)]
    blocks = schematic.blocks
    # TODO generalize to more general block properties
    # Longer term: remove and put a call to the modify model here
    if d.get("has_colour"):
        old_idm = most_common_idm(blocks.values())
        c = block_data.COLOR_BID_MAP.get(d["has_colour"])
        if c is not None:
            new_idm = random.choice(c)
            for l in blocks:
                if blocks[l] == old_idm:
                    blocks[l] = new_idm
    return list(blocks.items()), schematic.memid, tags


def interpret_schematic(
    interpreter, speaker, d, repeat_dict=None
) -> List[Tuple[List[Block], Optional[str], List[Tuple[str, str]]]]:
    """Return a list of 3-tuples, each with values:
    - the schematic blocks, list[(xyz, idm)]
    - a SchematicNode memid, or None
    - a list of (pred, val) tags
    """
    # hack, fixme in grammar/standardize.  sometimes the repeat is a sibling of action
    if repeat_dict is not None:
        repeat = cast(int, get_repeat_num(repeat_dict))
    else:
        repeat = cast(int, get_repeat_num(d))
    assert type(repeat) == int, "bad repeat={}".format(repeat)
    if "has_shape" in d:
        blocks, tags = interpret_shape_schematic(interpreter, speaker, d)
        return [(blocks, None, tags)] * repeat
    else:
        return [interpret_named_schematic(interpreter, speaker, d)] * repeat


def maybe_get_location_memory(interpreter, speaker, d):
    location_type = d.get("location_type", "SPEAKER_LOOK")
    if location_type == "REFERENCE_OBJECT" or d.get("reference_object") is not None:
        if d.get("relative_direction") == "BETWEEN":
            if d.get("reference_object_1"):
                mem1 = interpret_reference_object(
                    interpreter, speaker, d["reference_object_1"], loose_speakerlook=True
                )[0]
                mem2 = interpret_reference_object(
                    interpreter, speaker, d["reference_object_2"], loose_speakerlook=True
                )[0]
                mems = [mem1, mem2]
            else:
                mems = interpret_reference_object(
                    interpreter, speaker, d["reference_object"], limit=2, loose_speakerlook=True
                )
                if len(mems) < 2:
                    mem1 = None
                else:
                    mem1, mem2 = mems
            if not mem1:
                # TODO specify the ref object in the error message
                raise ErrorWithResponse("I don't know what you're referring to")
            loc = (np.add(mem1.get_pos(), mem2.get_pos())) / 2
            loc = (loc[0], loc[1], loc[2])
        else:
            mems = interpret_reference_object(interpreter, speaker, d["reference_object"])
            if len(mems) == 0:
                tags = set(tags_from_dict(d["reference_object"]))
                cands = interpreter.memory.get_recent_entities("Mobs")
                mems = [c for c in cands if any(set.intersection(set(c.get_tags()), tags))]
                if len(mems) == 0:
                    cands = interpreter.memory.get_recent_entities("BlockObjects")
                    mems = [c for c in cands if any(set.intersection(set(c.get_tags()), tags))]
                    if len(mems) == 0:
                        raise ErrorWithResponse("I don't know what you're referring to")
            assert len(mems) == 1, mems
            interpreter.memory.update_recent_entities(mems)
            mem = mems[0]
            loc = mem.get_pos()
            mems = [mem]
        return loc, mems
    return None, None


def interpret_location(interpreter, speaker, d, ignore_reldir=False) -> Tuple[XYZ, Any]:
    """Location dict -> coordinates, maybe ref obj memory
    Side effect:  adds mems to agent_memory.recent_entities
    if a reference object is interpreted;
    and loc to memory
    """
    mem = None
    location_type = d.get("location_type", "SPEAKER_LOOK")
    if location_type == "SPEAKER_LOOK":
        player_struct = interpreter.agent.perception_modules[
            "low_level"
        ].get_player_struct_by_name(speaker)
        loc = capped_line_of_sight(interpreter.agent, player_struct)

    elif location_type == "SPEAKER_POS":
        loc = pos_to_np(
            interpreter.agent.perception_modules["low_level"]
            .get_player_struct_by_name(speaker)
            .pos
        )

    elif location_type == "AGENT_POS":
        loc = pos_to_np(interpreter.agent.get_player().pos)

    elif location_type == "COORDINATES":
        loc = cast(XYZ, tuple(int(float(w)) for w in re.findall("[-0-9.]+", d["coordinates"])))
        if len(loc) != 3:
            logging.error("Bad coordinates: {}".format(d["coordinates"]))
            raise ErrorWithResponse("I don't understand what location you're referring to")
    else:
        loc, mems = maybe_get_location_memory(interpreter, speaker, d)
        mem = mems[0]
        if loc is None:
            raise ValueError("Can't handle Location type: {}".format(location_type))

    # handle relative direction
    reldir = d.get("relative_direction")
    if reldir is not None and not ignore_reldir:
        if reldir == "BETWEEN":
            pass  # loc already handled when getting mems above
        elif reldir == "INSIDE":
            if location_type == "REFERENCE_OBJECT":
                mem = mems[0]
                locs = heuristic_perception.find_inside(mem)
                if len(locs) == 0:
                    raise ErrorWithResponse("I don't know how to go inside there")
                else:
                    loc = locs[0]
                    mem = None
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
    return to_block_pos(loc), mem


def interpret_point_target(interpreter, speaker, d):
    if d.get("location") is None:
        # TODO other facings
        raise ErrorWithResponse("I am not sure where you want me to point")
    loc, mem = interpret_location(interpreter, speaker, d["location"])
    if mem is not None:
        return mem.get_point_at_target()
    else:
        return (loc[0], loc[1] + 1, loc[2], loc[0], loc[1] + 1, loc[2])


def number_from_span(span):
    # this will fail in many cases....
    words = span.split()
    degrees = None
    for w in words:
        try:
            degrees = int(w)
        except:
            pass
    if not degrees:
        try:
            degrees = word_to_num(span)
        except:
            pass
    return degrees


def interpret_facing(interpreter, speaker, d):
    current_pitch = interpreter.agent.get_player().look.pitch
    current_yaw = interpreter.agent.get_player().look.yaw
    if d.get("yaw_pitch"):
        span = d["yaw_pitch"]
        # for now assumed in (yaw, pitch) or yaw, pitch or yaw pitch formats
        yp = span.replace("(", "").replace(")", "").split()
        return {"head_yaw_pitch": (int(yp[0]), int(yp[1]))}
    elif d.get("yaw"):
        # for now assumed span is yaw as word or number
        w = d["yaw"].strip(" degrees").strip(" degree")
        return {"head_yaw_pitch": (word_to_num(w), current_pitch)}
    elif d.get("pitch"):
        # for now assumed span is pitch as word or number
        w = d["pitch"].strip(" degrees").strip(" degree")
        return {"head_yaw_pitch": (current_yaw, word_to_num(w))}
    elif d.get("relative_yaw"):
        # TODO in the task use turn angle
        if d["relative_yaw"].get("angle"):
            return {"relative_yaw": int(d["relative_yaw"]["angle"])}
        elif d["relative_yaw"].get("yaw_span"):
            span = d["relative_yaw"].get("yaw_span")
            left = "left" in span or "leave" in span  # lemmatizer :)
            degrees = number_from_span(span) or 90
            if degrees > 0 and left:
                print(-degrees)
                return {"relative_yaw": -degrees}
            else:
                print(degrees)
                return {"relative_yaw": degrees}
        else:
            pass
    elif d.get("relative_pitch"):
        if d["relative_pitch"].get("angle"):
            # TODO in the task make this relative!
            return {"relative_pitch": int(d["relative_pitch"]["angle"])}
        elif d["relative_pitch"].get("pitch_span"):
            span = d["relative_pitch"].get("pitch_span")
            down = "down" in span
            degrees = number_from_span(span) or 90
            if degrees > 0 and down:
                return {"relative_pitch": -degrees}
            else:
                return {"relative_pitch": degrees}
        else:
            pass
    elif d.get("location"):
        loc, _ = interpret_location(interpreter, speaker, d["location"])
        return {"head_xyz": loc}
    else:
        raise ErrorWithResponse("I am not sure where you want me to turn")


def interpret_stop_condition(interpreter, speaker, d) -> Optional[StopCondition]:
    if d.get("condition_type") == "NEVER":
        return NeverStopCondition(interpreter.agent)
    elif d.get("condition_type") == "ADJACENT_TO_BLOCK_TYPE":
        block_type = d["block_type"]
        bid, meta = get_block_type(block_type)
        return AgentAdjacentStopCondition(interpreter.agent, bid)
    else:
        return None


def get_holes(interpreter, speaker, location, limit=1, all_proximity=10) -> List[Tuple[XYZ, Hole]]:
    holes: List[Hole] = heuristic_perception.get_all_nearby_holes(interpreter.agent, location)
    candidates: List[Tuple[XYZ, Hole]] = [
        (to_block_pos(np.mean(hole[0], axis=0)), hole) for hole in holes
    ]
    if len(candidates) > 0:
        # NB(demiguo): by default, we fill the hole the player is looking at
        player_struct = interpreter.agent.perception_modules[
            "low_level"
        ].get_player_struct_by_name(speaker)
        centroid_hole = object_looked_at(interpreter.agent, candidates, player_struct, limit=limit)
        if centroid_hole is None or len(centroid_hole) == 0:
            # NB(demiguo): if there's no hole in front of the player, we will fill the nearest hole
            speaker_pos = (
                interpreter.agent.perception_modules["low_level"]
                .get_player_struct_by_name(speaker)
                .pos
            )
            speaker_pos = to_block_pos(pos_to_np(speaker_pos))
            if limit == "ALL":
                return list(
                    filter(lambda c: euclid_dist(c[0], speaker_pos) <= all_proximity, candidates)
                )
            else:
                candidates.sort(key=lambda c: euclid_dist(c[0], speaker_pos))
                return candidates[:limit]
        else:
            return centroid_hole
    else:
        return []


def get_mobs(interpreter, *tags) -> List[Tuple[XYZ, MobNode]]:
    """Return a list of (xyz, memory) tuples, filtered by tags"""
    mobs = interpreter.memory.get_mobs_tagged(*tags)
    return [(to_block_pos(mob.pos), mob) for mob in mobs]


def get_players(interpreter, *tags) -> List[Tuple[XYZ, PlayerNode]]:
    """Return a list of (xyz, memory) tuples, filtered by tags"""
    players = interpreter.memory.get_players_tagged(*tags)
    return [(to_block_pos(np.array((player.pos))), player) for player in players]


def get_objects(interpreter, *tags) -> List[Tuple[XYZ, ReferenceObjectNode]]:
    """Return a list of (xyz, memory) tuples, filtered by tags"""

    def post_process(m):
        if (
            m.__class__.__name__ == "BlockObjectNode"
            or m.__class__.__name__ == "ComponentObjectNode"
        ):
            m_pos = to_block_pos(np.mean(strip_idmeta(m.blocks.items()), axis=0))
        elif m.__class__.__name__ == "InstSegNode":
            m_pos = to_block_pos(np.mean(m.locs, axis=0))
        else:
            return None
        return (m_pos, m)

    mems = interpreter.memory.get_block_objects_with_tags(*tags)
    mems += interpreter.memory.get_component_objects_with_tags(*tags)
    mems += interpreter.memory.get_instseg_objects_with_tags(*tags)

    return [post_process(m) for m in mems]


def get_reference_objects(interpreter, *tags) -> List[Tuple[XYZ, ReferenceObjectNode]]:
    """Return a list of (xyz, memory) tuples encompassing all possible reference objects"""
    objs = cast(List[Tuple[XYZ, ReferenceObjectNode]], get_objects(interpreter, *tags))
    mobs = cast(List[Tuple[XYZ, ReferenceObjectNode]], get_mobs(interpreter, *tags))
    players = cast(List[Tuple[XYZ, ReferenceObjectNode]], get_players(interpreter, *tags))
    return objs + mobs + players


# TODO filter by INSIDE/AWAY/NEAR
def filter_by_sublocation(
    interpreter,
    speaker,
    candidates: List[Tuple[XYZ, T]],
    location: Dict,
    limit=1,
    all_proximity=10,
    loose=False,
) -> List[Tuple[XYZ, T]]:
    """Select from a list of candidate (xyz, object) tuples given a sublocation

    If limit == 'ALL', return all matching candidates

    Returns a list of (xyz, mem) tuples
    """

    # handle SPEAKER_LOOK separately due to slightly different semantics
    # (proximity to ray instead of point)
    if location.get("location_type") == "SPEAKER_LOOK":
        player_struct = interpreter.agent.perception_modules[
            "low_level"
        ].get_player_struct_by_name(speaker)
        return object_looked_at(
            interpreter.agent, candidates, player_struct, limit=limit, loose=loose
        )

    reldir = location.get("relative_direction")
    if reldir:
        if reldir == "INSIDE":
            if location.get("reference_object"):
                # this is ugly, should probably return from interpret_location...
                ref_mems = interpret_reference_object(
                    interpreter, speaker, location["reference_object"]
                )
                for l, candidate_mem in candidates:
                    if heuristic_perception.check_inside([candidate_mem, ref_mems[0]]):
                        return [(l, candidate_mem)]
            raise ErrorWithResponse("I can't find something inside that")
        elif reldir == "AWAY":
            raise ErrorWithResponse("I don't know which object you mean")
        elif reldir == "NEAR":
            pass  # fall back to no reference direction
        elif reldir == "BETWEEN":
            ref_loc, _ = interpret_location(interpreter, speaker, location)
            candidates.sort(key=lambda c: euclid_dist(c[0], ref_loc))
            return candidates[:limit]
        else:
            # reference object location, i.e. the "X" in "left of X"
            ref_loc, _ = interpret_location(interpreter, speaker, location, ignore_reldir=True)
            # relative direction, i.e. the "LEFT" in "left of X"
            reldir_vec = rotation.DIRECTIONS[reldir]

            # transform each object into the speaker look coordinate system,
            # and project onto the reldir vector
            look = (
                interpreter.agent.perception_modules["low_level"]
                .get_player_struct_by_name(speaker)
                .look
            )
            proj = [
                rotation.transform(np.array(l) - ref_loc, look.yaw, 0) @ reldir_vec
                for (l, _) in candidates
            ]

            # filter by relative dir, e.g. "left of Y"
            proj_cands = [(p, c) for (p, c) in zip(proj, candidates) if p > 0]

            # "the X left of Y" = the right-most X that is left of Y
            if limit == "ALL":
                limit = len(proj_cands)
            return [c for (_, c) in sorted(proj_cands, key=lambda p: p[0])][:limit]
    else:  # is it even possible to end up in this branch? FIXME?
        # no reference direction: choose the closest
        ref_loc, _ = interpret_location(interpreter, speaker, location, ignore_reldir=True)
        if limit == "ALL":
            return list(filter(lambda c: euclid_dist(c[0], ref_loc) <= all_proximity, candidates))
        else:
            candidates.sort(key=lambda c: euclid_dist(c[0], ref_loc))
            return candidates[:limit]
    return []  # this fixes flake but seems awful?


def get_repeat_arrangement(
    d, interpreter, speaker, schematic, repeat_num=-1, extra_space=1
) -> List[XYZ]:
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
            central_object = interpret_reference_object(
                interpreter, speaker, d["location"]["reference_object"], limit=1
            )
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


def get_repeat_num(d) -> Union[int, str]:
    if "repeat" in d:
        repeat_dict = d["repeat"]
        if repeat_dict["repeat_key"] == "FOR":
            try:
                return word_to_num(repeat_dict["repeat_count"])
            except:
                return 2  # TODO: dialogue instead of default?
        if repeat_dict["repeat_key"] == "ALL":
            return "ALL"
    return 1


def get_block_type(s) -> IDM:
    """string -> (id, meta)"""
    name_to_bid = minecraft_specs.get_block_data()["name_to_bid"]
    s_aug = s + " block"
    _, closest_match = min(
        [(name, id_meta) for (name, id_meta) in name_to_bid.items() if id_meta[0] < 256],
        key=lambda x: min(Levenshtein.distance(x[0], s), Levenshtein.distance(x[0], s_aug)),
    )

    return closest_match


def strip_prefix(s, pre):
    if s.startswith(pre):
        return s[len(pre) :]
    return s
