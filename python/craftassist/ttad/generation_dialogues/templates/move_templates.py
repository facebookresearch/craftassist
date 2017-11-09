# fmt: off
'''
Every template contains an ordered list of TemplateObjects.
TemplateObject is defined in template_objects.py

Move templates are written with respect to a Location and represent the intent
for the action: Move. This action specifies the location to which the
agent is expected to move to.

Things to note:
- RelativeDirectionTemplate when not followed by something signifies that RelativeDirection
is with respect to the agent / person who you are speaking to.

Examples:
[Move, ALittle, RelativeDirectionTemplate]
- move a bit to your left.
- move a little to the right.
- walk a little to the front.

[Move, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate]
- walk 5 steps to the right of that grey thing
- move fifty two steps to the left of the orange structure
'''


from template_objects import *

MOVE_WITH_CORRECTION = [
    ## Go there, to the rel_dir of the mob ##
    [[Human, Move, ThereTemplate],
     [HumanReplace, Move, RelativeDirectionTemplate, LocationMobTemplate]],
    [[Human, Move, ThereTemplate],
     [HumanReplace, Move, RelativeDirectionTemplate, BlockObjectThat]],
    [[Human, Move, ThereTemplate],
     [HumanReplace, Move, RelativeDirectionTemplate, BlockObjectThis]],

    [[Human, Move, ThereTemplateCoref],
     [HumanReplace, Move, RelativeDirectionTemplate, LocationMobTemplate]],
    [[Human, Move, ThereTemplateCoref],
    [HumanReplace, Move, RelativeDirectionTemplate, BlockObjectThat]],
    [[Human, Move, ThereTemplateCoref],
    [HumanReplace, Move, RelativeDirectionTemplate, BlockObjectThis]],

    [[Human, MoveSingle],
     [HumanReplace, Move, RelativeDirectionTemplate, LocationMobTemplate]],
    [[Human, MoveSingle],
     [HumanReplace, Move, RelativeDirectionTemplate, BlockObjectThat]],
    [[Human, MoveSingle],
     [HumanReplace, Move, RelativeDirectionTemplate, BlockObjectThis]],

    [[Human, MoveHere],
     [HumanReplace, Move, RelativeDirectionTemplate, LocationMobTemplate]],
    [[Human, MoveHere],
     [HumanReplace, Move, RelativeDirectionTemplate, BlockObjectThat]],
    [[Human, MoveHere],
     [HumanReplace, Move, RelativeDirectionTemplate, BlockObjectThis]],
    [[Human, MoveHereCoref],
     [HumanReplace, Move, RelativeDirectionTemplate, LocationMobTemplate]],
    [[Human, MoveHereCoref],
     [HumanReplace, Move, RelativeDirectionTemplate, BlockObjectThat]],
    [[Human, MoveHereCoref],
     [HumanReplace, Move, RelativeDirectionTemplate, BlockObjectThis]],
    [[Human, Move, RelativeDirectionTemplate],
     [HumanReplace, Move, RelativeDirectionTemplate, Until, ConditionTypeAdjacentBlockType]],
    [[Human, Move, ALittle, RelativeDirectionTemplate],
     [HumanReplace, Move, RelativeDirectionTemplate, Until, ConditionTypeAdjacentBlockType]],
    [[Human, Move, To, LocationBlockObjectTemplate],
     [HumanReplace, Move, RelativeDirectionTemplate, LocationBlockObjectTemplate]],
    [[Human, Move, To, BlockObjectThat],
     [HumanReplace, Move, RelativeDirectionTemplate, BlockObjectThat]],
    [[Human, Move, To, BlockObjectThis],
     [HumanReplace, Move, RelativeDirectionTemplate, BlockObjectThis]],
    [[Human, Move, To, LocationMobTemplate],
     [HumanReplace, Move, RelativeDirectionTemplate, LocationMobTemplate]],
    ]

MOVE_TEMPLATES = [
    ## One word command for Move ##
    [Human, MoveSingle],

    # Move with Location ##
    [Human, MoveHere],
    [Human, MoveHereCoref],
    [Human, MoveHere, ConditionTypeNever],
    [Human, Move, ThereTemplate],
    [Human, Move, ThereTemplateCoref],

    [Human, Move, RelativeDirectionTemplate],
    [Human, Move, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Move, RelativeDirectionTemplate, BlockObjectThis],
    [Human, Move, RelativeDirectionTemplate, StepsTemplate],
    [Human, Move, RelativeDirectionTemplate, Until, ConditionTypeAdjacentBlockType],
    [Human, Move, RelativeDirectionTemplate, LocationBlockObjectTemplate, Until, ConditionTypeAdjacentBlockType],
    [Human, Move, ALittle, RelativeDirectionTemplate],
    [Human, Move, ALittle, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Move, ALittle, RelativeDirectionTemplate, BlockObjectThis],
    [Human, Move, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Move, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Move, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],

    [Human, Move, To, CoordinatesTemplate],
    [Human, Move, To, LocationWord, CoordinatesTemplate],
    [Human, Move, To, LocationBlockObjectTemplate],
    [Human, Move, To, LocationMobTemplate],
    [Human, Move, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Move, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Move, RelativeDirectionTemplate, LocationMobTemplate],

    ## Other ways of saying Move ##
    [Human, Find, LocationMobTemplate],
    [Human, Move, To, Where, LocationMobTemplate, Is],

    # Follow Mob ##
    [Human, Move, LocationMobTemplate, ConditionTypeNever],
    [Human, Move, ThisTemplate, LocationMobTemplate, ConditionTypeNever],

    ## Move n steps m times ##
    [Human, Move, StepsTemplate, NTimes],
    [Human, Move, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Move, StepsTemplate, RelativeDirectionTemplate, BlockObjectThis],
    [Human, Move, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Move, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Move, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],

    ## Climb to the top of X ##
    [Human, Move, ClimbDirectionTemplate, LocationBlockObjectTemplate],
    ] + MOVE_WITH_CORRECTION
