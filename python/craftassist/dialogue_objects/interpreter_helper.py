"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import logging
import Levenshtein
import numpy as np
import random
import re
from typing import cast, List, Tuple, Union, Optional, Dict

from base_agent.dialogue_objects import ConfirmReferenceObject, SPEAKERLOOK
import block_data
import minecraft_specs
import heuristic_perception
import rotation
import size_words
from base_agent.memory_nodes import ReferenceObjectNode
from base_agent.stop_condition import StopCondition, NeverStopCondition
from mc_stop_condition import AgentAdjacentStopCondition
from .reference_object_helpers import compute_locations

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
    to_block_pos,
)
from word2number.w2n import word_to_num
from word_maps import SPECIAL_SHAPE_FNS, SPECIAL_SHAPES_CANONICALIZE


# FIXME!? maybe start using triples appropriately now?
def tags_from_dict(d):
    return [
        strip_prefix(tag, "the ")
        for key, tag in d.items()
        if key.startswith("has_") and isinstance(tag, str)
    ]


def get_special_reference_object(interpreter, speaker, S):
    # TODO/FIXME! add things to workspace memory
    if S == "SPEAKER_LOOK":
        player_struct = interpreter.agent.perception_modules[
            "low_level"
        ].get_player_struct_by_name(speaker)
        loc = capped_line_of_sight(interpreter.agent, player_struct)
        memid = interpreter.memory.add_location((int(loc[0]), int(loc[1]), int(loc[2])))
        mem = interpreter.memory.get_location_by_id(memid)

    elif S == "SPEAKER":
        p = interpreter.agent.perception_modules["low_level"].get_player_struct_by_name(speaker)
        mem = interpreter.memory.get_player_by_eid(p.entityId)

    elif S == "AGENT":
        p = interpreter.agent.get_player()
        mem = interpreter.memory.get_player_by_eid(p.entityId)

    elif type(S) is dict:
        coord_span = S["coordinates_span"]
        loc = cast(XYZ, tuple(int(float(w)) for w in re.findall("[-0-9.]+", coord_span)))
        if len(loc) != 3:
            logging.error("Bad coordinates: {}".format(coord_span))
            raise ErrorWithResponse("I don't understand what location you're referring to")
        memid = interpreter.memory.add_location((int(loc[0]), int(loc[1]), int(loc[2])))
        mem = interpreter.memory.get_location_by_id(memid)
    return mem


def interpret_reference_object(
    interpreter,
    speaker,
    d,
    only_voxels=False,
    only_physical=False,
    only_destructible=False,
    not_location=False,
    limit=1,
    loose_speakerlook=False,
) -> List[ReferenceObjectNode]:
    """this tries to find a ref obj memory matching the criteria from the
    ref_obj_dict
    """

    F = d.get("filters")
    special = d.get("special_reference")
    # F can be empty...
    assert (F is not None) or special, "no filters or special_reference sub-dicts {}".format(d)
    if special:
        mem = get_special_reference_object(interpreter, speaker, special)
        return [mem]

    if F.get("contains_coreference", "NULL") != "NULL":
        mem = F["contains_coreference"]
        if isinstance(mem, ReferenceObjectNode):
            return [mem]
        else:
            logging.error("bad coref_resolve -> {}".format(mem))

    if len(interpreter.progeny_data) == 0:
        tags = tags_from_dict(F)
        if only_voxels:
            tags.append("_voxel_object")
        if only_physical:
            tags.append("_physical_object")
        if only_destructible:
            tags.append("_destructible")
        # FIXME hack until memory_filters supprts "not"
        if not_location:
            tags.append("_not_location")
        # TODO Add ignore_player maybe?
        candidates = get_reference_objects(interpreter, *tags)
        if len(candidates) > 0:
            r = filter_by_sublocation(
                interpreter, speaker, candidates, d, limit=limit, loose=loose_speakerlook
            )
            return [mem for _, mem in r]
        else:
            # no candidates found; ask Clarification
            # TODO: move ttad call to dialogue manager and remove this logic
            interpreter.action_dict_frozen = True
            player_struct = interpreter.agent.perception_modules[
                "low_level"
            ].get_player_struct_by_name(speaker)
            tags = []
            if only_voxels:
                tags.append("_voxel_object")
            if only_physical:
                tags.append("_physical_object")
            if only_destructible:
                tags.append("_destructible")
            confirm_candidates = get_reference_objects(interpreter, *tags)
            objects = object_looked_at(
                interpreter.agent, confirm_candidates, player_struct, limit=1
            )
            if len(objects) == 0:
                raise ErrorWithResponse("I don't know what you're referring to")
            _, mem = objects[0]
            interpreter.provisional["object_mem"] = mem
            interpreter.provisional["F"] = F
            interpreter.dialogue_stack.append_new(ConfirmReferenceObject, mem)
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


def interpret_reference_location(interpreter, speaker, d):
    """
    Location dict -> coordinates of reference objc and maybe a list of ref obj
        memories.
    Side effect: adds mems to agent_memory.recent_entities
    """
    loose_speakerlook = False
    expected_num = 1
    if d.get("relative_direction") == "BETWEEN":
        loose_speakerlook = True
        expected_num = 2
        ref_obj_1 = d.get("reference_object_1")
        ref_obj_2 = d.get("reference_object_2")
        if ref_obj_1 and ref_obj_2:
            mem1 = interpret_reference_object(
                interpreter, speaker, ref_obj_1, loose_speakerlook=loose_speakerlook
            )[0]
            mem2 = interpret_reference_object(
                interpreter, speaker, ref_obj_2, loose_speakerlook=True
            )[0]
            if mem1 is None or mem2 is None:
                raise ErrorWithResponse("I don't know what you're referring to")
            mems = [mem1, mem2]
            interpreter.memory.update_recent_entities(mems)
            return mems

    ref_obj = d.get("reference_object", SPEAKERLOOK["reference_object"])
    mems = interpret_reference_object(
        interpreter, speaker, ref_obj, limit=expected_num, loose_speakerlook=loose_speakerlook
    )

    if len(mems) < expected_num:
        tags = set(tags_from_dict(ref_obj))
        cands = interpreter.memory.get_recent_entities("Mob")
        mems = [c for c in cands if any(set.intersection(set(c.get_tags()), tags))]

    if len(mems) < expected_num:
        cands = interpreter.memory.get_recent_entities("BlockObject")
        mems = [c for c in cands if any(set.intersection(set(c.get_tags()), tags))]

    if len(mems) < expected_num:
        raise ErrorWithResponse("I don't know what you're referring to")

    mems = mems[:expected_num]
    interpreter.memory.update_recent_entities(mems)
    # TODO: are there any memories where get_pos() doesn't return something?
    return mems


def interpret_point_target(interpreter, speaker, d):
    if d.get("location") is None:
        # TODO other facings
        raise ErrorWithResponse("I am not sure where you want me to point")
    # TODO: We might want to specifically check for BETWEEN/INSIDE, I'm not sure
    # what the +1s are in the return value
    mems = interpret_reference_location(interpreter, speaker, d["location"])
    loc, _ = compute_locations(interpreter, speaker, d, mems)
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
        mems = interpret_reference_location(interpreter, speaker, d["location"])
        loc, _ = compute_locations(interpreter, speaker, d, mems)
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


# TODO: This seems like a good candidate for cognition
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


def get_reference_objects(interpreter, *tags) -> List[Tuple[XYZ, ReferenceObjectNode]]:
    """Return a list of (xyz, memory) tuples encompassing all possible reference objects"""
    f = {"triples": [{"pred_text": "has_tag", "obj_text": tag} for tag in tags]}
    mems = interpreter.memory.get_reference_objects(f)
    return [(m.get_pos(), m) for m in mems]


# TODO filter by INSIDE/AWAY/NEAR
def filter_by_sublocation(
    interpreter,
    speaker,
    candidates: List[Tuple[XYZ, T]],
    d: Dict,
    limit=1,
    all_proximity=10,
    loose=False,
) -> List[Tuple[XYZ, T]]:
    """Select from a list of candidate (xyz, object) tuples given a sublocation

    If limit == 'ALL', return all matching candidates

    Returns a list of (xyz, mem) tuples
    """
    F = d.get("filters")
    assert F is not None, "no filters".format(d)
    location = F.get("location", SPEAKERLOOK)
    if limit == 1:
        limit = get_repeat_num(d)

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
                # this is ugly, should probably return from interpret_reference_location...
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
            mems = interpret_reference_location(interpreter, speaker, location)
            ref_loc, _ = compute_locations(interpreter, speaker, d, mems)
            candidates.sort(key=lambda c: euclid_dist(c[0], ref_loc))
            return candidates[:limit]
        else:
            # reference object location, i.e. the "X" in "left of X"
            mems = interpret_reference_location(interpreter, speaker, location)
            ref_loc = mems[0].get_pos()

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
        mems = interpret_reference_location(interpreter, speaker, location)
        ref_loc, _ = compute_locations(interpreter, speaker, d, mems)
        if limit == "ALL":
            return list(filter(lambda c: euclid_dist(c[0], ref_loc) <= all_proximity, candidates))
        else:
            candidates.sort(key=lambda c: euclid_dist(c[0], ref_loc))
            return candidates[:limit]
    return []  # this fixes flake but seems awful?


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


# TODO FILTERS!
def get_block_type(s) -> IDM:
    """string -> (id, meta)
    or  {"has_x": span} -> (id, meta) """

    name_to_bid = minecraft_specs.get_block_data()["name_to_bid"]
    if type(s) is str:
        s_aug = s + " block"
        _, closest_match = min(
            [(name, id_meta) for (name, id_meta) in name_to_bid.items() if id_meta[0] < 256],
            key=lambda x: min(Levenshtein.distance(x[0], s), Levenshtein.distance(x[0], s_aug)),
        )
    else:
        if "has_colour" in s:
            c = block_data.COLOR_BID_MAP.get(s["has_colour"])
            if c is not None:
                closest_match = random.choice(c)
        if "has_block_type" in s:
            _, closest_match = min(
                [(name, id_meta) for (name, id_meta) in name_to_bid.items() if id_meta[0] < 256],
                key=lambda x: min(
                    Levenshtein.distance(x[0], s), Levenshtein.distance(x[0], s_aug)
                ),
            )

    return closest_match


def strip_prefix(s, pre):
    if s.startswith(pre):
        return s[len(pre) :]
    return s
