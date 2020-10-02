from condition import (
    LinearExtentValue,
    LinearExtentAttribute,
    FixedValue,
    convert_comparison_value,
)
from base_util import ErrorWithResponse, number_from_span
from base_agent.memory_nodes import ReferenceObjectNode
from dialogue_object_utils import tags_from_dict


def interpret_span_value(interpreter, speaker, d, comparison_measure=None):
    num = number_from_span(d)
    if num:
        v = FixedValue(interpreter.agent, num)
        # always convert everything to internal units
        # FIXME handle this better
        v = convert_comparison_value(v, comparison_measure)
    else:
        v = FixedValue(interpreter.agent, d)
    return v


def maybe_specific_mem(interpreter, speaker, ref_obj_d):
    mem = None
    search_data = None
    if ref_obj_d.get("special_reference"):
        # this is a special ref object, not filters....
        cands = interpreter.subinterpret["reference_objects"](interpreter, speaker, ref_obj_d)
        return cands[0], None
    filters_d = ref_obj_d.get("filters", {})
    coref = filters_d.get("contains_coreference")
    if coref != "NULL":
        # this is a particular entity etc, don't search for the mem at check() time
        if isinstance(coref, ReferenceObjectNode):
            mem = coref
        else:
            cands = interpreter.subinterpret["reference_objects"](interpreter, speaker, ref_obj_d)
        if not cands:
            # FIXME fix this error
            raise ErrorWithResponse("I don't know which objects attribute you are talking about")
        # TODO if more than one? ask?
        else:
            mem = cands[0]
    else:
        # this object is only defined by the filters and might be different at different moments
        tags = tags_from_dict(filters_d)
        # make a function, reuse code with get_reference_objects FIXME
        search_data = [{"pred_text": "has_tag", "obj_text": tag} for tag in tags]

    return mem, search_data


def interpret_linear_extent(interpreter, speaker, d, force_value=False):
    location_data = {}
    default_frame = getattr(interpreter.agent, "default_frame") or "AGENT"
    frame = d.get("frame", default_frame)
    if frame == "SPEAKER":
        frame = speaker
    if type(frame) is dict:
        frame = frame.get("player_span", "unknown_player")
    if frame == "AGENT":
        location_data["frame"] = "AGENT"
    else:
        p = interpreter.agent.memory.get_player_by_name(frame)
        if p:
            location_data["frame"] = p.eid
        else:
            raise ErrorWithResponse("I don't understand in whose frame of reference you mean")
    location_data["relative_direction"] = d.get("relative_direction", "AWAY")
    # FIXME!!!! has_measure

    rd = d.get("source")
    fixed_role = "source"
    if not rd:
        rd = d.get("destination")
        fixed_role = "destination"
    mem, sd = maybe_specific_mem(interpreter, speaker, rd)
    L = LinearExtentAttribute(interpreter.agent, location_data, mem=mem, fixed_role=fixed_role)

    # TODO some sort of sanity check here, these should be rare:
    if (d.get("source") and d.get("destination")) or force_value:
        rd = d.get("destination")
        mem = None
        sd = None
        if rd:
            mem, sd = maybe_specific_mem(interpreter, speaker, rd["filters"])
        L = LinearExtentValue(interpreter.agent, L, mem=mem, search_data=sd)

    return L
