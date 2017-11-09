"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

"""This file contains functions to map strings representing abstract
sizes to ints and ranges"""

import numpy as np

RANGES = {
    "tiny": (1, 3),
    "small": (3, 4),
    "medium": (4, 6),
    "large": (6, 10),
    "huge": (10, 16),
    "gigantic": (16, 32),
}


MODIFIERS = ["very", "extremely", "really"]

WORD_SUBS = {"little": "small", "big": "large"}


def size_str_to_int(s):
    a, b = size_str_to_range(s)
    return np.random.randint(a, b)


def size_str_to_range(s):
    words = s.split()

    # replace words in WORD_SUBS
    for i in range(len(words)):
        if words[i] in WORD_SUBS:
            words[i] = WORD_SUBS[words[i]]

    is_modded = any(m in words for m in MODIFIERS)
    med_idx = list(RANGES.keys()).index("medium")

    for i, (word, rng) in enumerate(RANGES.items()):
        if word in words:
            if is_modded and i < med_idx:
                return list(RANGES.values())[i - 1]
            elif is_modded and i > med_idx:
                return list(RANGES.values())[i + 1]
            else:
                return list(RANGES.values())[i]

    return RANGES["medium"]


def size_int_to_str(x):
    raise NotImplementedError()
