We support the following types of dialogues in the V1 bot:
- HUMAN_GIVE_COMMAND
- GET_MEMORY
- PUT_MEMORY

The following actions are supported in the V1 bot:

- Build
- Copy
- Noop
- Spawn
- Resume
- Fill
- Destroy
- Move
- Undo
- Stop
- Dig
- FreeBuild
- Dance


The detailed action dictionary of each dialogue_type and action is given in the following subsections.

## Build Action ##
This is the action to Build a schematic at an optional location.

The Build action can have one of the following as its child:
- location only
- schematic only
- location and schematic both
- neither

```
{ "dialogue_type": "HUMAN_GIVE_COMMAND",
  "action_sequence" : [
    { "action_type" : 'BUILD',
      "location" : {
          "location_type" : COORDINATES / REFERENCE_OBJECT / AGENT_POS / SPEAKER_POS / SPEAKER_LOOK,
          "steps" : span,
          "contains_coreference" : "yes",
          "relative_direction" : 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK'/ 'AWAY'
                                  / 'INSIDE' / 'NEAR' / 'OUTSIDE' / 'BETWEEN',
          "coordinates" : span,
          "reference_object" : {
              "repeat" : {
                  "repeat_key" : 'FOR'/ 'ALL'
                  "repeat_count" : span,
                  "repeat_dir": 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK' / 'AROUND'
              }
              "has_name" : span,
              "has_size" : span,
              "has_colour" : span,
              "contains_coreference" : "yes",
              "location" : {
                  "contains_coreference" : "yes",
                  "location_type" : COORDINATES / AGENT_POS / SPEAKER_POS / SPEAKER_LOOK,
                  "coordinates" : span
               } } },
       "schematic" : {
          "repeat" : {
             "repeat_key" : 'FOR'
             "repeat_count" : span,
             "repeat_dir": 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK' / 'AROUND' / 'SURROUND'
           }
          "has_block_type" : span,
          "has_name": span,
          "has_size" : span,
          "has_orientation" : span,
          "has_thickness" : span,
          "has_colour" : span,
          "has_height" : span,
          "has_length" : span,
          "has_radius" : span,
          "has_slope" : span,
          "has_width" : span,
          "has_base" : span,
          "has_distance" : span,
        },
        "repeat" : {
           "repeat_key" : 'FOR'
           "repeat_count" : span,
           "repeat_dir": 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK' / 'AROUND' / 'SURROUND'
         },
         "replace": True
       }
      ]
}
```

## Copy Action ##
This is the action to copy a block object to an optional location. The copy action is represented as a "Build" with an optional reference_object in the tree.

Copy action can have one the following as its child:
- reference_object
- reference_object and location
- neither

```
{ "dialogue_type": "HUMAN_GIVE_COMMAND",
  "action_sequence" : [
    { "action_type" : 'BUILD',
      "location" : {
          "location_type" : COORDINATES / REFERENCE_OBJECT / AGENT_POS / SPEAKER_POS / SPEAKER_LOOK,
          "steps" : span,
          "contains_coreference" : "yes",
          "relative_direction" : 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK'/
                                 'AWAY' / 'INSIDE' / 'NEAR' / 'OUTSIDE' / 'BETWEEN',
          "coordinates" : span,
          "reference_object" : {
              "repeat" : {
                  "repeat_key" : 'FOR'/ 'ALL'
                  "repeat_count" : span,
                  "repeat_dir": 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK' / 'AROUND'
              }
              "has_name" : span,
              "has_size" : span,
              "has_colour" : span,
              "contains_coreference" : "yes",
              "location" : {
                  "contains_coreference" : "yes",
                  "location_type" : COORDINATES / AGENT_POS / SPEAKER_POS / SPEAKER_LOOK,
                  "coordinates" : span
               } } },
       "reference_object" : {
          "has_size" : span,
          "has_colour" : span,
          "has_name" : span,
          "contains_coreference" : "yes",
          "repeat" : {
            "repeat_key" : 'FOR'/ 'ALL'
            "repeat_count" : span,
            "repeat_dir": 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK' / 'AROUND'
          }
          "location" : {
              "location_type" : COORDINATES / AGENT_POS / SPEAKER_POS / SPEAKER_LOOK,
              "coordinates" : span
          } },
       "repeat" : {
         "repeat_key" : 'FOR'
         "repeat_count" : span,
         "repeat_dir": 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK' / 'AROUND'
       },
       "replace": True
       }
      ]
}
```

## Noop Action ##
This action indicates no operation should be performed.

```
{ "dialogue_type": "NOOP"}
```

## Spawn Action ##
This action indicates that the specified object should be spawned in the environment.

Spawn action has the following child:
- reference_object

```
{ "dialogue_type": "HUMAN_GIVE_COMMAND",
  "action_sequence" : [
    { "action_type" : 'SPAWN'
      "reference_object" : {
          "repeat" : {
            "repeat_key" : 'FOR'
            "repeat_count" : span,
            "repeat_dir": 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK' / 'AROUND'
          }
          "has_name" : span,
        },
      "location" : {
          "location_type" : COORDINATES / REFERENCE_OBJECT / AGENT_POS / SPEAKER_POS / SPEAKER_LOOK,
          "steps" : span,
          "contains_coreference" : "yes",
          "relative_direction" : 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK'/
                                 'AWAY' / 'INSIDE' / 'NEAR' / 'OUTSIDE' / 'BETWEEN',
          "coordinates" : span,
          "reference_object" : {
              "repeat" : {
                  "repeat_key" : 'FOR'/ 'ALL'
                  "repeat_count" : span,
                  "repeat_dir": 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK' / 'AROUND'
              }
              "has_name" : span,
              "has_size" : span,
              "has_colour" : span,
              "contains_coreference" : "yes",
              "location" : {
                  "contains_coreference" : "yes",
                  "location_type" : COORDINATES / AGENT_POS / SPEAKER_POS / SPEAKER_LOOK,
                  "coordinates" : span
               } } },
        "repeat" : {
          "repeat_key" : 'FOR'
          "repeat_count" : span,
          "repeat_dir": 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK' / 'AROUND'
        },
        "replace": True
      }
    ]
}
```

## Resume Action ##
This action indicates that the previous action should be resumed.

```
{ "dialogue_type": "HUMAN_GIVE_COMMAND",
  "action_sequence" : [
    { "action_type" : 'RESUME',
      "target_action_type": span
    }
  ]
}

```

## Fill Action ##
This action states that a hole / negative shape at an optional location needs to be filled up.

Fill action can have one of the following as its child:
- location
- nothing

```
{ "dialogue_type": "HUMAN_GIVE_COMMAND",
  "action_sequence" : [
    { "action_type" : 'FILL',
      "has_block_type" : span,
      "reference_object" : {
          "location" : {
              "location_type" : COORDINATES / REFERENCE_OBJECT / AGENT_POS / SPEAKER_POS / SPEAKER_LOOK,
              "steps" : span,
              "contains_coreference" : "yes",
              "relative_direction" : 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK'/
                                     'AWAY' / 'INSIDE' / 'NEAR' / 'OUTSIDE' / 'BETWEEN',
              "coordinates" : span,
              "reference_object" : {
                  "repeat" : {
                      "repeat_key" : 'FOR'/ 'ALL'
                      "repeat_count" : span,
                      "repeat_dir": 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK' / 'AROUND'
                  }
                  "has_name" : span,
                  "has_size" : span,
                  "has_colour" : span,
                  "contains_coreference" : "yes",
                  "location" : {
                      "contains_coreference" : "yes",
                      "location_type" : COORDINATES / AGENT_POS / SPEAKER_POS / SPEAKER_LOOK,
                      "coordinates" : span
                   } } },
         "has_colour" : span,
         "contains_coreference" : "yes",
         "has_name" : span,
         "has_size" : span,
         "repeat" : {
           "repeat_key" : 'FOR' / 'ALL',
           "repeat_count" : span,
           "repeat_dir": 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK' / 'AROUND'
         } },
      "replace": True
      }
    ]
}
```

## Destroy Action ##
This action indicates the intent to destroy a block object at an optional location.

Destroy action can have on of the following as the child:
- reference_object
- nothing

```
{ "dialogue_type": "HUMAN_GIVE_COMMAND",
  "action_sequence" : [
    { "action_type" : 'DESTROY',
      "reference_object" : {
          "location" : {
              "location_type" : COORDINATES / REFERENCE_OBJECT / AGENT_POS / SPEAKER_POS / SPEAKER_LOOK,
              "steps" : span,
              "contains_coreference" : "yes",
              "relative_direction" : 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK'/
                                     'AWAY' / 'INSIDE' / 'NEAR' / 'OUTSIDE' / 'BETWEEN',
              "coordinates" : span,
              "reference_object" : {
                  "repeat" : {
                      "repeat_key" : 'FOR'/ 'ALL'
                      "repeat_count" : span,
                      "repeat_dir": 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK' / 'AROUND'
                  }
                  "has_name" : span,
                  "has_size" : span,
                  "has_colour" : span,
                  "contains_coreference" : "yes",
                  "location" : {
                      "contains_coreference" : "yes",
                      "location_type" : COORDINATES / AGENT_POS / SPEAKER_POS / SPEAKER_LOOK,
                      "coordinates" : span
                   } } },
         "has_colour" : span,
         "contains_coreference" : "yes",
         "has_name" : span,
         "has_size" : span,
         "repeat" : {
           "repeat_key" : 'FOR' / 'ALL',
           "repeat_count" : span,
           "repeat_dir": 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK' / 'AROUND'
         } },
    "replace": True
    }
  ]
}
```

## Move Action ##
This action states that the agent should move to the specified location.

Move action can have one of the following as its child:
- location
- stop_condition (stop moving when a condition is met)
- location and stop_condition
- neither

```
{ "dialogue_type": "HUMAN_GIVE_COMMAND",
  "action_sequence" : [
    { "action_type" : 'MOVE',
      "location" : {
          "location_type" : COORDINATES / REFERENCE_OBJECT / AGENT_POS / SPEAKER_POS / SPEAKER_LOOK,
          "steps" : span,
          "contains_coreference" : "yes",
          "relative_direction" : 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK'/
                                 'AWAY' / 'INSIDE' / 'NEAR'/ 'OUTSIDE' / 'BETWEEN',
          "coordinates" : span,
          "reference_object" : {
              "repeat" : {
                  "repeat_key" : 'FOR'/ 'ALL'
                  "repeat_count" : span,
                  "repeat_dir": 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK' / 'AROUND'
              }
              "has_name" : span,
              "has_size" : span,
              "has_colour" : span,
              "contains_coreference" : "yes",
              "location" : {
                  "contains_coreference" : "yes",
                  "location_type" : COORDINATES / AGENT_POS / SPEAKER_POS / SPEAKER_LOOK,
                  "coordinates" : span
               } } },
        "stop_condition" : {
            "condition_type" : 'ADJACENT_TO_BLOCK_TYPE' / 'NEVER',
            "block_type" : span,
            "condition_span" : span,
        },
        "repeat" : {
          "repeat_key" : 'FOR',
          "repeat_count" : span,
          "repeat_dir": 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK' / 'AROUND'
        },
        "replace": True
        }
    ]
}
```

## Undo Action ##
This action states the intent to revert the specified action, if any.

Undo action can have on of the following as its child:
- undo_action
- nothing (meaning : undo the last action)

```
{ "dialogue_type": "HUMAN_GIVE_COMMAND",
  "action_sequence" : [
    { "action_type" : 'UNDO',
      "target_action_type" : span
    }
  ]
}
```

## Stop Action ##
This action indicates stop.

```
{ "dialogue_type": "HUMAN_GIVE_COMMAND",
  "action_sequence" : [
    { "action_type" : 'STOP',
      "target_action_type": span
    }
  ]
}
```

## Dig Action ##
This action represents the intent to dig a hole / negative shape of optional dimensions at an optional location.

Dig action can have one of the following as its child:
- nothing
- location
- stop_condition
- location and stop_condition
and / or has_size_, has_length_, has_depth_, has_width_

```
{ "dialogue_type": "HUMAN_GIVE_COMMAND",
  "action_sequence" : [
    { "action_type" : 'DIG',
      "location" : {
          "location_type" : COORDINATES / REFERENCE_OBJECT / AGENT_POS / SPEAKER_POS / SPEAKER_LOOK,
          "steps" : span,
          "contains_coreference" : "yes",
          "relative_direction" : 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK'/
                                 'AWAY' / 'INSIDE' / 'NEAR' / 'OUTSIDE' / 'BETWEEN',
          "coordinates" : span,
          "reference_object" : {
              "repeat" : {
                  "repeat_key" : 'FOR'/ 'ALL'
                  "repeat_count" : span,
                  "repeat_dir": 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK' / 'AROUND'
              }
              "has_name" : span,
              "has_size" : span,
              "has_colour" : span,
              "contains_coreference" : "yes",
              "location" : {
                  "contains_coreference" : "yes",
                  "location_type" : COORDINATES / AGENT_POS / SPEAKER_POS / SPEAKER_LOOK,
                  "coordinates" : span
               } } },
       "schematic" : {
          "repeat" : {
             "repeat_key" : 'FOR'
             "repeat_count" : span,
             "repeat_dir": 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK' / 'AROUND'
           }
           "has_size" : span,
           "has_length" : span,
           "has_depth" : span,
           "has_width" : span,
        },
       "stop_condition" : {
           "condition_type" : 'ADJACENT_TO_BLOCK_TYPE' / 'NEVER',
           "block_type" : span
       },
      "replace": True  
      }
    ]
}
```

## FreeBuild action ##
This action represents that the agent should complete an already existing half-finished block object, using its mental model.

FreeBuild action can have one of the following as its child:
- reference_object only
- reference_object and location

```
{ "dialogue_type": "HUMAN_GIVE_COMMAND",
  "action_sequence" : [
    { "action_type" : 'FREEBUILD',
      "reference_object" : {
          "location" : {
              "location_type" : COORDINATES / REFERENCE_OBJECT / AGENT_POS / SPEAKER_POS / SPEAKER_LOOK,
              "steps" : span,
              "contains_coreference" : "yes",
              "relative_direction" : 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/
                                     'BACK'/ 'AWAY' / 'INSIDE' / 'NEAR' /
                                     'OUTSIDE' / 'BETWEEN',
              "coordinates" : span,
              "reference_object" : {
                  "repeat" : {
                      "repeat_key" : 'FOR'/ 'ALL'
                      "repeat_count" : span,
                      "repeat_dir": 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK' / 'AROUND'
                  }
                  "has_name" : span,
                  "has_size" : span,
                  "has_colour" : span,
                  "contains_coreference" : "yes",
                  "location" : {
                      "contains_coreference" : "yes",
                      "location_type" : COORDINATES / AGENT_POS / SPEAKER_POS / SPEAKER_LOOK,
                      "coordinates" : span
                   } } },
          "has_size" : span,
          "has_colour" : span,
          "has_name" : span,
          "contains_coreference" : "yes",
          "repeat" : {
            "repeat_key" : 'FOR'/'ALL',
            "repeat_count" : span,
            "repeat_dir": 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK' / 'AROUND'
          } },
      "location" : {
          "location_type" : COORDINATES / REFERENCE_OBJECT / AGENT_POS / SPEAKER_POS / SPEAKER_LOOK,
          "steps" : span,
          "contains_coreference" : "yes",
          "relative_direction" : 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/
                                 'BACK'/ 'AWAY' / 'INSIDE' / 'NEAR' / 'OUTSIDE' / 'BETWEEN',
          "coordinates" : span,
          "reference_object" : {
              "repeat" : {
                  "repeat_key" : 'FOR'/ 'ALL'
                  "repeat_count" : span,
                  "repeat_dir": 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK' / 'AROUND'
              }
              "has_name" : span,
              "has_size" : span,
              "has_colour" : span,
              "contains_coreference" : "yes",
              "location" : {
                  "contains_coreference" : "yes",
                  "location_type" : COORDINATES / AGENT_POS / SPEAKER_POS / SPEAKER_LOOK,
                  "coordinates" : span
               } } },
      "replace": True
      }
    ]
}
```

## Dance Action ##
This action provides information to the agent to do a dance.

Dance action can have one of the following as its child:
- location
- stop_condition (stop dancing when a condition is met)
- location and stop_condition
- repeat
- nothing

```
{ "dialogue_type": "HUMAN_GIVE_COMMAND",
  "action_sequence" : [
    { "action_type" : 'DANCE',
      "location" : {
          "location_type" : COORDINATES / REFERENCE_OBJECT / AGENT_POS / SPEAKER_POS / SPEAKER_LOOK,
          "steps" : span,
          "contains_coreference" : "yes",
          "relative_direction" : 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK'/
                                 'AWAY' / 'INSIDE' / 'NEAR' / 'OUTSIDE' / 'BETWEEN',
          "coordinates" : span,
          "reference_object" : {
              "repeat" : {
                  "repeat_key" : 'FOR'/ 'ALL'
                  "repeat_count" : span,
                  "repeat_dir": 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK' / 'AROUND'
              }
              "has_name" : span,
              "has_size" : span,
              "has_colour" : span,
              "contains_coreference" : "yes",,
              "location" : {
                  "contains_coreference" : "yes",
                  "location_type" : COORDINATES / AGENT_POS / SPEAKER_POS / SPEAKER_LOOK,
                  "coordinates" : span
               } } },
      "stop_condition" : {
          "condition_type" : NEVER,
      },
      "repeat" : {
        "repeat_key" : 'FOR',
        "repeat_count" : span,
        "repeat_dir" : 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK' / 'AROUND'
      },
      "replace": True
      }
    ]
}
```

## GetMemory dialogue ##
This dialogue type provides information that can be used to filter
memory objects from the memory, and the answer_type to extract.

GetMemory dialogue can have the following as its child:
- filters
- answer_type

```
{
  "dialogue_type": "GET_MEMORY",
  "filters": {
    "temporal": CURRENT,
    "type": "ACTION" / "AGENT" / "REFERENCE_OBJECT",
    "action_type": BUILD / DESTROY / DIG / FILL / SPAWN / MOVE
    "reference_object" : {
        "location" : {
            "location_type" : COORDINATES / REFERENCE_OBJECT / AGENT_POS / SPEAKER_POS / SPEAKER_LOOK,
            "contains_coreference" : "yes",
            "relative_direction" : 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK'/
                                   'AWAY' / 'NEAR' / 'INSIDE' / 'OUTSIDE' / 'BETWEEN',
            "coordinates" : span,
            "reference_object" : {
                "has_name" : span,
                "has_size" : span,
                "has_colour" : span,
                "contains_coreference" : "yes",
                "location" : {
                    "contains_coreference" : "yes",
                    "location_type" : COORDINATES / AGENT_POS / SPEAKER_POS / SPEAKER_LOOK,
                    "coordinates" : span
                 } } },
        "has_size" : span,
        "has_colour" : span,
        "has_name" : span,
        "coref_resolve": span,
       },
  },
  "answer_type": "TAG" / "EXISTS" ,
  "tag_name" : 'has_name' / 'has_size' / 'has_colour' / 'action_name' /
              'action_reference_object_name' / 'move_target' / 'location' ,
  "replace": true
}
```

## PutMemory dialogue ##
This dialogue type provides information that can be used to filter
memory objects from the memory, and write an info_type to the memory.

GetMemory dialogue can have the following as its child:
- filters
- info_type

```
{
  "dialogue_type": "PUT_MEMORY",
  "filters": {
    "reference_object" : {
      "location" : {
          "location_type" : COORDINATES / REFERENCE_OBJECT / AGENT_POS / SPEAKER_POS / SPEAKER_LOOK,
          "steps" : span,
          "contains_coreference" : "yes",
          "relative_direction" : 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK'/
                                 'AWAY' / 'INSIDE' / 'NEAR' / 'OUTSIDE' / 'BETWEEN',
          "coordinates" : span,
          "reference_object" : {
              "repeat" : {
                  "repeat_key" : 'FOR'/ 'ALL'
                  "repeat_count" : span,
                  "repeat_dir": 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK' / 'AROUND'
              }
              "has_name" : span,
              "has_size" : span,
              "has_colour" : span,
              "contains_coreference" : "yes",
              "location" : {
                  "contains_coreference" : "yes",
                  "location_type" : COORDINATES / AGENT_POS / SPEAKER_POS / SPEAKER_LOOK,
                  "coordinates" : span
               } } },
      "has_size" : span,
      "has_colour" : span,
      "has_name" : span,
      "contains_coreference" : "yes",
      "repeat" : {
        "repeat_key" : 'FOR'/'ALL',
        "repeat_count" : span,
        "repeat_dir": 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK' / 'AROUND'
      }
     },
  },
  "upsert" : {
      "memory_data": {
        "memory_type": "REWARD" / "TRIPLE",
        "reward_value": "POSITIVE" / "NEGATIVE",
        "has_tag" : span,
        "has_colour": span,
        "has_size": span
      } }
}
```
