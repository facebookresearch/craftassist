"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

from preprocess import word_tokenize
from ttad.generation_dialogues.generate_dialogue import action_type_map  # Dict[str, Class]
from ttad.generation_dialogues.templates.templates import template_map  # Dict[str, List[Template]]

from typing import Dict, List, Tuple, Sequence


def adtt(d: Dict) -> Tuple[str, Dict]:
    """Return a string that would produce the action dict `d`

    THIS IS NOT FULLY WORKING! Some relative directions (and maybe other stuff)
    are not yet working. In the mean time, also return the generated action
    dict gen_d which corresponds to the returned text, and whose
    relative_direction values may be different than the input d.

    d is post-process_span (i.e. its span values are replaced with strings)
    d is pre-coref_resolve (i.e. its coref_resolve values are strings, not
        memory objects or keywords)

    """
    if d["dialogue_type"] != "HUMAN_GIVE_COMMAND":
        raise NotImplementedError("can't handle {}".format(d["dialogue_type"]))

    action_type = d["action"]["action_type"]  # e.g. "MOVE"
    action_type = action_type[0].upper() + action_type[1:].lower()  # e.g. "Move"

    for template in template_map[action_type]:
        dialogue, gen_d = generate_from_template(action_type, template)
        recurse_remove_keys(gen_d, ["has_attribute"])
        if len(dialogue) != 1:
            continue
        if dicts_match(d, gen_d):
            return replace_spans(dialogue[0], gen_d, d), gen_d

    raise ValueError("No matching template found for {}".format(d))


def replace_spans(text: str, gen_d: Dict, d: Dict) -> str:
    """Replace words in text with spans from d"""

    words = word_tokenize(text).split()

    # compile list of spans to replace via recursive search
    replaces = []
    to_consider = [(gen_d, d)]
    while len(to_consider) > 0:
        cur_gen_d, cur_d = to_consider.pop()
        for k in cur_gen_d.keys():
            if type(cur_d[k]) == dict:
                to_consider.append((cur_gen_d[k], cur_d[k]))
            elif type(cur_d[k]) == str and cur_d[k].upper() != cur_d[k]:
                replaces.append((cur_gen_d[k], cur_d[k]))

    # replace each span in words
    replaces.sort(key=lambda r: r[0][1][0], reverse=True)  # sort by L of span
    for (sentence_idx, (L, R)), s in replaces:
        assert sentence_idx == 0
        words = words[:L] + word_tokenize(s).split() + words[(R + 1) :]

    return " ".join(words)


def generate_from_template(action_type: str, template: List) -> Tuple[List[str], Dict]:
    cls = action_type_map[action_type.lower()]
    node = cls.generate(template)
    dialogue = node.generate_description()
    d = node.to_dict()
    return dialogue, d


def dicts_match(
    d: Dict,
    e: Dict,
    ignore_values_for_keys: Sequence[str] = ["relative_direction"],
    ignore_keys: Sequence[str] = ["has_attribute"],
) -> bool:
    if (set(d.keys()) - set(ignore_keys)) != (set(e.keys()) - set(ignore_keys)):
        return False

    for k, v in d.items():
        if type(v) == dict and not dicts_match(v, e[k]):
            return False

        # allow values of certain keys to differ (e.g. relative_direction)
        # allow spans (lowercase strings) to differ
        if (
            k not in ignore_keys
            and k not in ignore_values_for_keys
            and type(v) == str
            and v == v.upper()
            and v != e[k]
        ):
            return False

    return True


def recurse_remove_keys(d: Dict, keys: Sequence[str]):
    # remove keys from dict
    for x in keys:
        if x in d:
            del d[x]

    # recurse
    for k, v in d.items():
        if type(v) == dict:
            recurse_remove_keys(v, keys)


if __name__ == "__main__":
    d = {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action": {
            "action_type": "BUILD",
            "schematic": {"has_name": "barn"},
            "location": {
                "location_type": "REFERENCE_OBJECT",
                "relative_direction": "LEFT",
                "reference_object": {"has_name": "boat house"},
            },
        },
    }
    t = adtt(d)
    print(t)
