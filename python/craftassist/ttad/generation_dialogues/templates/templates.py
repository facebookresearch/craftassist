"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

"""This file picks a template for a given action, at random.

The templates use template_objects as their children to help construct a sentence
and the dictionary.

TemplateObject is defined in template_objects.py
Each template captures how to phrase the intent. The intent is defined by the action
type.
"""
import copy
import random

from template_objects import *
from build_templates import *
from move_templates import *
from dig_templates import *
from destroy_templates import *
from copy_templates import *
from undo_templates import *
from fill_templates import *
from spawn_templates import *
from freebuild_templates import *
from dance_templates import *
from get_memory_templates import *
from put_memory_templates import *
from stop_templates import *
from resume_templates import *

template_map = {
    "Move": MOVE_TEMPLATES,
    "Build": BUILD_TEMPLATES,
    "Destroy": DESTROY_TEMPLATES,
    "Dig": DIG_TEMPLATES,
    "Copy": COPY_TEMPLATES,
    "Undo": UNDO_TEMPLATES,
    "Fill": FILL_TEMPLATES,
    "Spawn": SPAWN_TEMPLATES,
    "Freebuild": FREEBUILD_TEMPLATES,
    "Dance": DANCE_TEMPLATES,
    "GetMemory": GET_MEMORY_TEMPLATES,
    "PutMemory": PUT_MEMORY_TEMPLATES,
    "Stop": STOP_TEMPLATES,
    "Resume": RESUME_TEMPLATES,
}


def get_template(template_key, node, template=None):
    """Pick a random template, given the action."""
    template_name = template_map[template_key]
    if template is None:
        template = random.choice(template_name)
    template = copy.deepcopy(template)

    if not any(isinstance(i, list) for i in template):
        template = [template]

    for i, t in enumerate(template):
        for j, templ in enumerate(t):
            if type(templ) != str:
                template[i][j] = templ(node=node)

    return template
