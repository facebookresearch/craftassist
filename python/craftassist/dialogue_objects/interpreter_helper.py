import logging
import Levenshtein
import numpy as np
import random
import re
from typing import cast, List, Tuple, Union, Optional, Dict, Any

import snowballstemmer

from .dialogue_object import ConfirmReferenceObject
import block_data
import minecraft_specs
import perception
import rotation
import shapes
import size_words
from memory import ObjectNode, MobNode, ReferenceObjectNode
from stop_condition import StopCondition, NeverStopCondition, AgentAdjacentStopCondition
from util import (
    Block,
    Hole,
    IDM,
    T,
    XYZ,
    capped_line_of_sight,
    euclid_dist,
    object_looked_at,
    pos_to_np,
    strip_idmeta,
    to_block_center,
    to_block_pos,
    ErrorWithResponse,
    NextDialogueStep,
)
from word2number.w2n import word_to_num
from word_maps import SPECIAL_SHAPE_FNS, SPECIAL_SHAPES_CANONICALIZE

stemmer = snowballstemmer.stemmer("english")


def interpret_reference_object(
    interpreter, speaker, d, ignore_mobs=False, limit=1
) -> List[ReferenceObjectNode]:

    if d.get("coref_resolve", "NULL") != "NULL":
        mem = d["coref_resolve"]
        if isinstance(mem, ReferenceObjectNode):
            return [mem]
        else:
            logging.error("bad coref_resolve -> {}".format(mem))

    if len(interpreter.progeny_data) == 0:
        tags = [
            stemmer.stemWord(tag.lstrip("the "))
            for key, tag in d.items()
            if key.startswith("has_") and isinstance(tag, str)
        ]
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
            r = filter_by_sublocation(interpreter, speaker, candidates, location_d, limit=limit)
            return [mem for _, mem in r]
        else:
            # no candidates found; ask Clarification
            # TODO: move ttad call to dialogue manager and remove this logic
            interpreter.action_dict_frozen = True

            player = interpreter.memory.get_player_struct_by_name(speaker)
            confirm_candidates = get_objects(interpreter)  # no tags
            objects = object_looked_at(interpreter.agent, confirm_candidates, player, limit=1)
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
    speaker, d, shapename=None
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
        "has_orientation",
        "has_distance",
        "has_base",
    ]

    attrs = {key[4:-1]: word_to_num(d[key]) for key in numeric_keys if key in d}

    if "has_size" in d:
        attrs["size"] = interpret_size(d["has_size"])

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
            stemmed_val = stemmer.stemWord(val)
            tags.append((key, stemmed_val))

    return SPECIAL_SHAPE_FNS[shape](**attrs), tags


def interpret_size(text) -> Union[int, List[int]]:
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
    stemmed_name = stemmer.stemWord(name)
    shapename = SPECIAL_SHAPES_CANONICALIZE.get(name) or SPECIAL_SHAPES_CANONICALIZE.get(
        stemmed_name
    )
    if shapename:
        blocks, tags = interpret_shape_schematic(speaker, d, shapename=shapename)
        return blocks, None, tags

    schematic = interpreter.memory.get_schematic_by_name(name)
    if schematic is None:
        schematic = interpreter.memory.get_schematic_by_name(stemmed_name)
        if schematic is None:
            raise ErrorWithResponse("I don't know what you're referring to")
    tags = [(p, v) for (_, p, v) in interpreter.memory.get_triples(subj=schematic.memid)]
    return list(schematic.blocks.items()), schematic.memid, tags


def interpret_schematic(
    interpreter, speaker, d
) -> List[Tuple[List[Block], Optional[str], List[Tuple[str, str]]]]:
    """Return a list of 3-tuples, each with values:
    - the schematic blocks, list[(xyz, idm)]
    - a SchematicNode memid, or None
    - a list of (pred, val) tags
    """
    repeat = cast(int, get_repeat_num(d))
    assert type(repeat) == int, "bad repeat={}".format(repeat)
    if "has_shape" in d:
        blocks, tags = interpret_shape_schematic(speaker, d)
        return [(blocks, None, tags)] * repeat
    else:
        return [interpret_named_schematic(interpreter, speaker, d)] * repeat


def interpret_location(interpreter, speaker, d, ignore_reldir=False) -> XYZ:
    """Location dict -> coordinates
    Side effect:  adds mems to agent_memory.recent_entities
    if a reference object is interpreted;
    and loc to memory
    """
    location_type = d.get("location_type", "SPEAKER_LOOK")
    if location_type == "REFERENCE_OBJECT":
        mems = interpret_reference_object(interpreter, speaker, d["reference_object"])
        if len(mems) == 0:
            raise ErrorWithResponse("I don't know what you're referring to")
        assert len(mems) == 1, mems
        interpreter.memory.update_recent_entities(mems)
        mem = mems[0]
        loc = mem.get_pos()

    elif location_type == "SPEAKER_LOOK":
        player = interpreter.memory.get_player_struct_by_name(speaker)
        loc = capped_line_of_sight(interpreter.agent, player)

    elif location_type == "SPEAKER_POS":
        loc = pos_to_np(interpreter.memory.get_player_struct_by_name(speaker).pos)

    elif location_type == "AGENT_POS":
        loc = pos_to_np(interpreter.agent.get_player().pos)

    elif location_type == "COORDINATES":
        loc = cast(XYZ, tuple(int(float(w)) for w in re.findall("[-0-9.]+", d["coordinates"])))
        if len(loc) != 3:
            logging.error("Bad coordinates: {}".format(d["coordinates"]))
            raise ErrorWithResponse("I don't understand what location you're referring to")

    else:
        raise ValueError("Can't handle Location type: {}".format(location_type))

    # handle relative direction
    reldir = d.get("relative_direction")
    if reldir is not None and not ignore_reldir:
        if reldir == "INSIDE":
            if location_type == "REFERENCE_OBJECT":
                locs = perception.find_inside(mem)
                if len(locs) == 0:
                    raise ErrorWithResponse("I don't know how to go inside there")
                else:
                    loc = locs[0]
        elif reldir == "AWAY":
            apos = pos_to_np(interpreter.agent.get_player().pos)
            dir_vec = (apos - loc) / np.linalg.norm(apos - loc)
            num_steps = word_to_num(d.get("steps", "5"))
            loc = num_steps * np.array(dir_vec) + to_block_center(loc)
        elif reldir == "NEAR":
            pass
        else:  # LEFT, RIGHT, etc...
            reldir_vec = rotation.DIRECTIONS[reldir]
            look = interpreter.memory.get_player_struct_by_name(speaker).look
            # this should be an inverse transform so we set inverted=True
            dir_vec = rotation.transform(reldir_vec, look.yaw, 0, inverted=True)
            num_steps = word_to_num(d.get("steps", "5"))
            loc = num_steps * np.array(dir_vec) + to_block_center(loc)

    # if steps without relative direction
    elif "steps" in d:
        num_steps = word_to_num(d.get("steps", "5"))
        loc = to_block_center(loc) + [0, 0, num_steps]
    return to_block_pos(loc)


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
    holes: List[Hole] = perception.get_all_nearby_holes(interpreter.agent, location)
    candidates: List[Tuple[XYZ, Hole]] = [
        (to_block_pos(np.mean(hole[0], axis=0)), hole) for hole in holes
    ]
    if len(candidates) > 0:
        # NB(demiguo): by default, we fill the hole the player is looking at
        player = interpreter.memory.get_player_struct_by_name(speaker)
        centroid_hole = object_looked_at(interpreter.agent, candidates, player, limit=limit)
        if centroid_hole is None or len(centroid_hole) == 0:
            # NB(demiguo): if there's no hole in front of the player, we will fill the nearest hole
            speaker_pos = interpreter.memory.get_player_struct_by_name(speaker).pos
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


def get_objects(interpreter, *tags) -> List[Tuple[XYZ, ObjectNode]]:
    """Return a list of (xyz, memory) tuples, filtered by tags"""

    def post_process(objs):
        obj_poss = [to_block_pos(np.mean(strip_idmeta(b.blocks.items()), axis=0)) for b in objs]
        return list(zip(obj_poss, objs))

    funcs = [
        interpreter.memory.get_block_objects_with_tags,
        interpreter.memory.get_component_objects_with_tags,
    ]
    return [p for func in funcs for p in post_process(func(*tags))]


def get_reference_objects(interpreter, *tags) -> List[Tuple[XYZ, ReferenceObjectNode]]:
    """Return a list of (xyz, memory) tuples encompassing all possible reference objects"""
    objs = cast(List[Tuple[XYZ, ReferenceObjectNode]], get_objects(interpreter, *tags))
    mobs = cast(List[Tuple[XYZ, ReferenceObjectNode]], get_mobs(interpreter, *tags))
    return objs + mobs


# TODO filter by INSIDE/AWAY/NEAR
def filter_by_sublocation(
    interpreter,
    speaker,
    candidates: List[Tuple[XYZ, T]],
    location: Dict,
    limit=1,
    all_proximity=10,
) -> List[Tuple[XYZ, T]]:
    """Select from a list of candidate (xyz, object) tuples given a sublocation

    If limit == 'ALL', return all matching candidates

    Returns a list of (xyz, mem) tuples
    """

    # handle SPEAKER_LOOK separately due to slightly different semantics
    # (proximity to ray instead of point)
    if location.get("location_type") == "SPEAKER_LOOK":
        player = interpreter.memory.get_player_struct_by_name(speaker)
        return object_looked_at(interpreter.agent, candidates, player, limit=limit)

    reldir = location.get("relative_direction")
    if reldir:  ###THIS IS UNTESTED revisit on new ttad model!!!! TODO
        if reldir == "INSIDE":
            if location.get("reference_object"):
                # this is ugly, should probably return from interpret_location...
                ref_mems = interpret_reference_object(
                    interpreter, speaker, location["reference_object"]
                )
                for l, candidate_mem in candidates:
                    if perception.check_inside([candidate_mem, ref_mems[0]]):
                        return [(l, candidate_mem)]
            raise ErrorWithResponse("I can't find something inside that")
        elif reldir == "AWAY":
            raise ErrorWithResponse("I don't know which object you mean")
        elif reldir == "NEAR":
            pass  # fall back to no reference direction
        else:
            # reference object location, i.e. the "X" in "left of X"
            ref_loc = interpret_location(interpreter, speaker, location, ignore_reldir=True)
            # relative direction, i.e. the "LEFT" in "left of X"
            reldir_vec = rotation.DIRECTIONS[reldir]

            # transform each object into the speaker look coordinate system,
            # and project onto the reldir vector
            look = interpreter.memory.get_player_struct_by_name(speaker).look
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
    else:
        # no reference direction: choose the closest
        if limit == "ALL":
            return list(filter(lambda c: euclid_dist(c[0], ref_loc) <= all_proximity, candidates))
        else:
            candidates.sort(key=lambda c: euclid_dist(c[0], ref_loc))
            return candidates[:limit]
    return []  # this fixes flake but seems awful?


def process_spans(d, words):
    for k, v in d.items():
        if type(v) == dict:
            process_spans(v, words)
            continue
        try:
            sentence, (L, R) = v
            if sentence != 0:
                raise NotImplementedError("Must update process_spans for multi-string inputs")
            assert 0 <= L <= R <= (len(words) - 1)
        except ValueError:
            continue
        except TypeError:
            continue

        d[k] = " ".join(words[L : (R + 1)])


def coref_resolve(memory, d: Dict):
    """Walk the entire action dict "d" and replace coref_resolve values

    Possible substitutions:
    - a keyword like "SPEAKER_POS"
    - a MemoryNode object
    - "NULL"

    Assumes spans have been substituted.
    """
    items_to_add: List[Tuple[Dict, Tuple[str, Any]]] = []

    for k, v in d.items():
        if k == "location":
            # v is a location dict
            for k_, v_ in v.items():
                if k_ == "coref_resolve":
                    val = "SPEAKER_POS" if v_ == "here" else "SPEAKER_LOOK"
                    items_to_add.append((v, ("location_type", val)))
        elif k == "reference_object":
            # v is a reference object dict
            for k_, v_ in v.items():
                if k_ == "coref_resolve":
                    if v_ in ["that", "this"]:
                        if "location" in v:
                            # overwrite the location_type without blowing away the rest of the dict
                            items_to_add.append((v["location"], ("location_type", "SPEAKER_LOOK")))
                        else:
                            # no location dict -- create one
                            items_to_add.append(
                                (v, ("location", {"location_type": "SPEAKER_LOOK"}))
                            )
                        v[k_] = "NULL"
                    else:
                        mems = memory.get_recent_entities("BlockObjects")
                        if len(mems) == 0:
                            v[k_] = "NULL"
                        else:
                            v[k_] = mems[0]

        if type(v) == dict:
            coref_resolve(memory, v)

    for d_, (k, v) in items_to_add:
        d_[k] = v


def get_repeat_arrangement(
    d, interpreter, speaker, schematic, repeat_num=-1, extra_space=1
) -> List[XYZ]:
    shapeparams = {}
    # eventually fix this to allow number based on shape
    shapeparams["N"] = repeat_num
    shapeparams["extra_space"] = extra_space
    direction_name = d.get("repeat", {}).get("repeat_dir", "FRONT")
    if direction_name != "AROUND":
        reldir_vec = rotation.DIRECTIONS[direction_name]
        look = interpreter.memory.get_player_struct_by_name(speaker).look
        # this should be an inverse transform so we set inverted=True
        dir_vec = rotation.transform(reldir_vec, look.yaw, 0, inverted=True)
        shapeparams["orient"] = dir_vec
        offsets = shapes.arrange("line", schematic, shapeparams)
    else:
        # TODO vertical "around"
        shapeparams["orient"] = "xy"
        shapeparams["encircled_object_radius"] = 1
        if d.get("central_object") is not None:
            central_object = interpret_reference_object(
                interpreter, speaker, d["central_object"], limit=1
            )
            # FIXME: .blocks is unsafe, object could be Mob. Ignoring for now.
            central_object = central_object[0].blocks  # type: ignore
            bounds = shapes.get_bounds(central_object)
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
