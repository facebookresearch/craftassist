"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import numpy as np
import os
import pickle

PATH = os.path.join(os.path.dirname(__file__), "../../minecraft_specs")
assert os.path.isdir(PATH), (
    "Not found: "
    + PATH
    + "\n\nDid you follow the instructions at "
    + "https://github.com/fairinternal/minecraft#getting-started"
)


_BLOCK_DATA = None


def get_block_data():
    global _BLOCK_DATA
    if _BLOCK_DATA is None:
        _BLOCK_DATA = _pickle_load("block_images/block_data")
    return _BLOCK_DATA


_COLOUR_DATA = None


def get_colour_data():
    global _COLOUR_DATA
    if _COLOUR_DATA is None:
        _COLOUR_DATA = _pickle_load("block_images/color_data")
    return _COLOUR_DATA


_PROP_DATA = None


def get_prop_data():
    global _PROP_DATA
    if _PROP_DATA is None:
        _PROP_DATA = _pickle_load("block_images/prop_data")
    return _PROP_DATA


def get_bid_to_colours():
    bid_to_name = get_block_data()["bid_to_name"]
    name_to_colors = get_colour_data()["name_to_colors"]
    bid_to_colors = {}
    for item in bid_to_name.keys():
        name = bid_to_name[item]
        if name in name_to_colors:
            color = name_to_colors[name]
            bid_to_colors[item] = color
    return bid_to_colors


def get_schematics(limit=-1):
    """Return a list of {'schematic': npy array, 'tags': tags, 'name': names} dicts"""
    schem_dir = os.path.join(PATH, "schematics")
    s = []
    for fname in os.listdir(schem_dir):
        fullname = os.path.join(schem_dir, fname)
        if not os.path.isdir(fullname):
            with open(fullname, "rb") as f:
                schematic_premem = pickle.load(f)
            assert type(schematic_premem["schematic"]) is np.ndarray
            s.append(schematic_premem)
    if limit > 0:
        s = np.random.permute(s)[:limit]
    return s


def _pickle_load(relpath):
    with open(os.path.join(PATH, relpath), "rb") as f:
        return pickle.load(f)
