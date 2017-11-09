"""
Copyright (c) Facebook, Inc. and its affiliates.

This file contains template objects associated with Dig action.
"""
import random

from generate_utils import *
from tree_components import *
from .template_object import *

#####################
### DIG TEMPLATES ###
#####################
dig_shapes = ["hole", "cave", "mine", "tunnel"]


"""This template object picks the shape of what will be dug"""


class DigSomeShape(TemplateObject):
    def __init__(self, node, template_attr):
        shape_type = DigShapeAny if pick_random(0.8) else DigShapeHole
        self._child = shape_type(node=node, template_attr=template_attr)

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return self._child.generate_description(arg_index=arg_index, index=index)


"""This template object represents specific shapes. Meant to generate direct
commands like : make a hole , dig a mine"""


class DigShapeHole(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        template_name = get_template_names(self, templ_index)
        plural = False
        phrase = random.choice(dig_shapes)

        if "RepeatCount" in template_name:
            phrase = make_plural(random.choice(dig_shapes))
            plural = True
        if not plural and (template_name[index - 1] not in ["DigDimensions", "DigAbstractSize"]):
            phrase = random.choice([phrase, prepend_a_an(phrase)])

        return phrase


"""This template object covers a variety of dig shape types and is more general
than DigShapeHole. It can also lead to generations like: 'dig down until you hit bedrock'
"""


class DigShapeAny(TemplateObject):
    def generate_description(self, arg_index=0, index=0, previous_text=None, templ_index=0):
        template_name = get_template_names(self, templ_index)

        if "RepeatCount" in template_name:
            phrase = make_plural(random.choice(dig_shapes))
        elif "DownTo" in template_name:
            phrase = random.choice(
                [random.choice(dig_shapes + ["grass"]), "a " + random.choice(dig_shapes)]
            )
        elif template_name[index - 1] in ["DigDimensions", "DigAbstractSize"]:
            phrase = random.choice(dig_shapes)
        elif index + 1 < len(template_name) and template_name[index + 1] == "NumBlocks":
            phrase = random.choice(
                [
                    random.choice(dig_shapes + ["under ground", "grass"]),
                    "a " + random.choice(dig_shapes),
                ]
            )
        else:
            phrase = random.choice(
                [
                    random.choice(
                        dig_shapes
                        + ["ground", "into ground", "under ground", "under grass", "grass", "down"]
                    ),
                    "a " + random.choice(dig_shapes),
                ]
            )

        return phrase


"""This template object assigns the dimensions: length, width and depth for
what needs to be dug."""


class DigDimensions(TemplateObject):
    def add_generate_args(self, index=0, templ_index=0):
        self.node.has_length = random.choice(self.template_attr.get("length", range(2, 15)))
        self.node.has_width = random.choice(self.template_attr.get("width", range(15, 30)))
        self.node.has_depth = (
            random.choice(self.template_attr.get("depth", range(30, 45)))
            if pick_random()
            else None
        )

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        template_name = get_template_names(self, templ_index)
        sizes = [self.node.has_length]

        if self.node.has_width:
            sizes.append(self.node.has_width)
        if self.node.has_depth:
            sizes.append(self.node.has_depth)
        out_size = random.choice([" x ".join(map(str, sizes)), " by ".join(map(str, sizes))])

        if ("RepeatCount" in template_name) or ("OfDimensions" in template_name):
            return out_size

        return "a " + out_size


"""This template object assigns an abstract size for the shape that needs
to be dug."""


class DigAbstractSize(TemplateObject):
    def add_generate_args(self, index=0, templ_index=0):
        self._size_description = random.choice(
            ABSTRACT_SIZE + ["deep", "very deep", "really deep"]
        )
        self.node.has_size = self._size_description

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        template_names = get_template_names(self, templ_index)
        if "RepeatCount" in template_names:
            return self._size_description
        phrase = random.choice([self._size_description, prepend_a_an(self._size_description)])
        return phrase


DIG_SHAPE_TEMPLATES = [DigSomeShape, DigShapeHole, DigShapeAny, DigDimensions, DigAbstractSize]
