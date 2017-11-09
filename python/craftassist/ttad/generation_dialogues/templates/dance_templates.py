"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

# fmt: off
'''
Every template contains an ordered list of TemplateObjects.
TemplateObject is defined in template_objects.py

Dance templates are written with an optional location and stop condition.

Examples:
[Human, DanceSingle]
- do a dance
- dance

[Human, DanceSingle, ConditionTypeNever],
- keep dancing
- dance until I tell you to stop
'''


from template_objects import *

DANCE_WITH_CORRECTION = [
    [[Human, DanceSingle],
     [HumanReplace, Dance, AroundString]]
]

DANCE_TEMPLATES = [
    ## Dance single word ##
    [Human, DanceSingle],
    [Human, DanceSingle, ConditionTypeNever],

    ## Walk around X ##
    [Human, Dance, AroundString, LocationBlockObjectTemplate],

    ## Move around X clockwise / anticlockwise ##
    [Human, Dance, AroundString, LocationBlockObjectTemplate, RelativeDirectionTemplate],

    ## move around X clockwise/anticlockwise n times ##
    [Human, Dance, AroundString, LocationBlockObjectTemplate, NTimes],
    [Human, Dance, AroundString, LocationBlockObjectTemplate, RelativeDirectionTemplate, NTimes]
] + DANCE_WITH_CORRECTION
