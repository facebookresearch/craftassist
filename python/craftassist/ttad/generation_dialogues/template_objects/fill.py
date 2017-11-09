"""
This file contains template objects associated with Fill action.
"""
import random

from generate_utils import *
from tree_components import *
from .template_object import *
from .dig import *

#####################
## FILL TEMPLATES ###
#####################


class FillShape(TemplateObject):
    """This template object repesents the shape/ thing that needs to be filled."""

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        template_names = get_template_names(self, templ_index)
        if any(x in ["RepeatCount", "RepeatAll"] for x in template_names):
            return make_plural(random.choice(dig_shapes))
        return random.choice(dig_shapes)


# Note: this is for "fill that mine" , no coref resolution needed
class FillObjectThis(TemplateObject):
    """This template object repesents that the thing to be filled is where the speaker
    is looking."""

    def add_generate_args(self, index=0, templ_index=0):
        phrases = ["this", "that"]
        template_names = get_template_names(self, templ_index)

        if any(x in ["RepeatCount", "RepeatAll"] for x in template_names):
            phrases = ["these", "those"]

        self._word = random.choice(phrases)
        self.node._location_args["coref_resolve"] = self._word

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return self._word


class FillBlockType(TemplateObject):
    def add_generate_args(self, index=0, templ_index=0):
        self.node.has_block_type = random.choice(BLOCK_TYPES)

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return self.node.has_block_type


class UseFill(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        phrase = random.choice(["use", "fill using", "fill with"])
        return phrase
