"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

LOCATION_RADIO = [
    {"text": "Not specified", "key": None},
    {
        "text": "Where the speaker is looking (e.g. 'that thing', 'over there')",
        "key": "SPEAKER_LOOK",
    },
    {"text": "Where the speaker is standing (e.g. 'here', 'by me')", "key": "SpeakerPos"},
    {
        "text": "Where the assistant is standing (e.g. 'by you', 'where you are')",
        "key": "AGENT_POS",
    },
    {
        "text": "To a specific object or area, or somewhere relative to other object(s)",
        "key": "REFERENCE_OBJECT",
        "next": [
            {"text": "What other object(s) or area?", "key": "has_name", "span": True},
            {
                "text": "Where in relation to the other object(s)?",
                "key": "relative_direction",
                "radio": [
                    {"text": "Left", "key": "LEFT"},
                    {"text": "Right", "key": "RIGHT"},
                    {"text": "Above", "key": "UP"},
                    {"text": "Below", "key": "DOWN"},
                    {"text": "In front", "key": "FRONT"},
                    {"text": "Behind", "key": "BACK"},
                    {"text": "Away from", "key": "AWAY"},
                    {"text": "Nearby", "key": "NEAR"},
                    {"text": "Exactly at", "key": "EXACT"},
                ],
            },
        ],
    },
]

REF_OBJECT_OPTIONALS = [
    {
        "text": "What is the building material?",
        "key": "reference_object.has_block_type",
        "span": True,
        "optional": True,
    },
    {
        "text": "What is the color?",
        "key": "reference_object.has_colour",
        "span": True,
        "optional": True,
    },
    {
        "text": "What is the size?",
        "key": "reference_object.has_size",
        "span": True,
        "optional": True,
    },
    {
        "text": "What is the width?",
        "key": "reference_object.has_width",
        "span": True,
        "optional": True,
    },
    {
        "text": "What is the height?",
        "key": "reference_object.has_height",
        "span": True,
        "optional": True,
    },
    {
        "text": "What is the depth?",
        "key": "reference_object.has_depth",
        "span": True,
        "optional": True,
    },
]


Q_ACTION = {
    "text": 'What action is being instructed? If multiple separate actions are being instructed (e.g. "do X and then do Y"), select "Multiple separate actions"',
    "key": "action_type",
    "add_radio_other": False,
    "radio": [
        # BUILD
        {
            "text": "Build or create something",
            "key": "BUILD",
            "next": [
                {
                    "text": "Is this a copy or duplicate of an existing object?",
                    "key": "COPY",
                    "radio": [
                        # COPY
                        {
                            "text": "Yes",
                            "key": "yes",
                            "next": [
                                {
                                    "text": "What object should be copied?",
                                    "key": "reference_object.has_name",
                                    "span": True,
                                },
                                *REF_OBJECT_OPTIONALS,
                                {
                                    "text": "What is the location of the object to be copied?",
                                    "key": "location",
                                    "radio": LOCATION_RADIO,
                                },
                            ],
                        },
                        # BUILD
                        {
                            "text": "No",
                            "key": "no",
                            "next": [
                                {
                                    "text": "What should be built?",
                                    "key": "schematic.has_name",
                                    "span": True,
                                },
                                {
                                    "text": "What is the building material?",
                                    "key": "schematic.has_block_type",
                                    "span": True,
                                    "optional": True,
                                },
                                {
                                    "text": "What is the size?",
                                    "key": "schematic.has_size",
                                    "span": True,
                                    "optional": True,
                                },
                                {
                                    "text": "What is the width?",
                                    "key": "schematic.has_width",
                                    "span": True,
                                    "optional": True,
                                },
                                {
                                    "text": "What is the height?",
                                    "key": "schematic.has_height",
                                    "span": True,
                                    "optional": True,
                                },
                                {
                                    "text": "What is the depth?",
                                    "key": "schematic.has_depth",
                                    "span": True,
                                    "optional": True,
                                },
                                {
                                    "text": "Is the assistant being asked to...",
                                    "key": "FREEBUILD",
                                    "add_radio_other": False,
                                    "radio": [
                                        {
                                            "text": "Build a complete, specific object",
                                            "key": "BUILD",
                                        },
                                        {
                                            "text": "Help complete or finish an existing object",
                                            "key": "FREEBUILD",
                                        },
                                    ],
                                },
                            ],
                        },
                    ],
                },
                {"text": "Where should it be built?", "key": "location", "radio": LOCATION_RADIO},
            ],
        },
        # MOVE
        {
            "text": "Move somewhere",
            "key": "MOVE",
            "next": [
                {
                    "text": "Where should the assistant move to?",
                    "key": "location",
                    "radio": LOCATION_RADIO,
                }
            ],
        },
        # DESTROY
        {
            "text": "Destroy, remove, or kill something",
            "key": "DESTROY",
            "next": [
                {
                    "text": "What object should the assistant destroy?",
                    "key": "reference_object.has_name",
                    "span": True,
                },
                *REF_OBJECT_OPTIONALS,
                {
                    "text": "What is the location of the object to be removed?",
                    "key": "location",
                    "radio": LOCATION_RADIO,
                },
            ],
        },
        # DIG
        {
            "text": "Dig a hole",
            "key": "DIG",
            "next": [
                {
                    "text": "Where should the hole be dug?",
                    "key": "location",
                    "radio": LOCATION_RADIO,
                },
                {"text": "What is the size?", "key": "has_size", "span": True, "optional": True},
                {"text": "What is the width?", "key": "has_width", "span": True, "optional": True},
                {
                    "text": "What is the height?",
                    "key": "has_height",
                    "span": True,
                    "optional": True,
                },
                {"text": "What is the depth?", "key": "has_depth", "span": True, "optional": True},
            ],
        },
        # FILL
        {
            "text": "Fill a hole",
            "key": "FILL",
            "next": [
                {
                    "text": "Where should the hole be dug?",
                    "key": "location",
                    "radio": LOCATION_RADIO,
                },
                *REF_OBJECT_OPTIONALS,
            ],
        },
        # TAG
        {
            "text": "Assign a description, name, or tag to an object",
            "key": "TAG",
            "tooltip": "e.g. 'That thing is fluffy' or 'The blue building is my house'",
            "next": [
                {
                    "text": "What is the description, name, or tag being assigned?",
                    "key": "tag",
                    "span": True,
                },
                {
                    "text": "What object is being assigned a description, name, or tag?",
                    "key": "reference_object",
                },
                *REF_OBJECT_OPTIONALS,
                {
                    "text": "What is the location of the object to be described, named, or tagged?",
                    "key": "location",
                    "radio": LOCATION_RADIO,
                },
            ],
        },
        # STOP
        {
            "text": "Stop current action",
            "key": "STOP",
            "next": [
                {
                    "text": "Is this a command to stop a particular action?",
                    "key": "target_action_type",
                    "radio": [
                        {"text": "Building", "key": "BUILD"},
                        {"text": "Moving", "key": "MOVE"},
                        {"text": "Destroying", "key": "DESTROY"},
                        {"text": "Digging", "key": "DIG"},
                        {"text": "Filling", "key": "FILL"},
                    ],
                }
            ],
        },
        # RESUME
        {"text": "Resume previous action", "key": "RESUME"},
        # UNDO
        {"text": "Undo previous action", "key": "UNDO"},
        # ANSWER QUESTION
        {
            "text": "Answer a question",
            "key": "ANSWER",
            "tooltip": "e.g. 'How many trees are there?' or 'Tell me how deep that tunnel goes'",
        },
        # OTHER ACTION NOT LISTED
        {
            "text": "Another action not listed here",
            "key": "OtherAction",
            "tooltip": "The sentence is a command, but not one of the actions listed here",
            "next": [
                {
                    "text": "What object (if any) is the target of this action? e.g. for the sentence 'Sharpen this axe', select the word 'axe'",
                    "key": "reference_object.has_name",
                    "span": True,
                },
                *REF_OBJECT_OPTIONALS,
                {
                    "text": "Where should the action take place?",
                    "key": "location",
                    "radio": LOCATION_RADIO,
                },
            ],
        },
        # NOT ACTION
        {
            "text": "This sentence is not a command or request to do something",
            "key": "NOOP",
            "tooltip": "e.g. 'Yes', 'Hello', or 'What a nice day it is today'",
        },
        # MULTIPLE ACTIONS
        {
            "text": "Multiple separate actions",
            "key": "COMPOSITE_ACTION",
            "tooltip": "e.g. 'Build a cube and then run around'. Do not select this for a single repeated action, e.g. 'Build 5 cubes'",
        },
    ],
}


REPEAT_DIR = {
    "text": "In which direction should the action be repeated?",
    "key": "repeat_dir",
    "radio": [
        {"text": "Not specified", "key": None},
        {"text": "Forward", "key": "FRONT"},
        {"text": "Backward", "key": "BACK"},
        {"text": "Left", "key": "LEFT"},
        {"text": "Right", "key": "RIGHT"},
        {"text": "Up", "key": "UP"},
        {"text": "Down", "key": "DOWN"},
    ],
}


Q_ACTION_LOOP = {
    "text": "How many times should this action be performed?",
    "key": "loop",
    "radio": [
        {"text": "Just once, or not specified", "key": None},
        {
            "text": "Repeatedly, a specific number of times",
            "key": "ntimes",
            "next": [{"text": "How many times?", "span": True, "key": "repeat_for"}],
        },
        {
            "text": "Repeatedly, once for each object",
            "key": "repeat_all",
            "tooltip": "e.g. 'Destroy the red blocks', or 'Build a shed in front of each house'",
        },
        {
            "text": "Repeated forever",
            "key": "forever",
            "tooltip": "e.g. 'Keep building railroad tracks in that direction' or 'Collect diamonds until I tell you to stop'",
        },
    ],
}
