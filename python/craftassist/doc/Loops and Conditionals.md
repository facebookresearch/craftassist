## Background

We want to handle sentences like:

- "dig down until you hit bedrock"
- "destroy all the blue objects"
- "follow me around"
- "build three houses over there"
- "build a tree in front of every house"
- "destroy everything"
- "make 5 copies of these"
- "create copies of every spherical shell"
- "help me mine a few holes"

## Proposal: Add new keys to the action dict

An `Action` has the same format as before along with some additional keys as described below.

Depending on the loop type, we add new keys to the dict:

- `stop_condition` to express infinite loops or loops conditioned on block types.
  This key will always be a child of the parent action, and has value as a dict. e.g. `{"Move": {"location": {"location_type": "SpeakerPos"}, "stop_condition": {"condition_type": "Never"}}}`
- `repeat_all` to express: for all objects (e.g. `reference_object` or `block_object`) that satisfy the condition, run the loop.
  This key only has one value: `"ALL"` and is nested inside the object.
- `repeat_for` to express: run the loop n times.
  This key has a span value (where values extracted from the span can be an int, num_to_word(int), "some" and "a few") and is nested inside: an object (e.g. `reference_object`, `block_object`, or `schematic`) , or an action (e.g. `Dig`).


A `StopCondition` has a key `condition_type` whose value is one of:

```
{
  "condition_type": {"Never", "AdjacentToBlockType"}
}
```
We can keep expanding the condition types as we come across more examples.

`StopCondition` has other keys depending on the `condition_type`:

```
{
  "condition_type": "AdjacentToBlockType"
  "block_type": [L, R]
}
```

## Examples


"dig down until you hit bedrock"
```
{
  "Dig": {
    "stop_condition": {
      "condition_type": "AdjacentToBlockType",
      "block_type": [5, 5]
    }
  }
}
```

"destroy all the blue objects"
```
{
 "Destroy": {
   "block_object": {
     "has_colour_": [3, 3],
     "repeat_all": "ALL"
     }
  }
}
```

"follow me around"
```
{
  "Move": {
    "location": {"location_type": "SpeakerPos"},
    "stop_condition": {
      "condition_type": "Never"
    }
  }
}
```

"build three houses over there"
```
{
  "Build": {
    "location": {"location_type": "SpeakerLook"},
    "schematic": {
      "has_name_": [2, 2],
      "repeat_for": [1, 1]
      }
  }
}
```

"build a tree in front of every house"
```
{
  "Build": {
    "location": {
      "relative_direction": "FRONT",
      "location_type": "BlockObject",
      "reference_object": {
        "has_name_": [7, 7],
        "repeat_all": "ALL"
      }
     },
    "schematic": {"has_name_": [2, 2]}
  }
}
```

"destroy everything"
```
{
  "Destroy": {
    "block_object": {
      "repeat_all": "ALL"
    }
  }
}
```

"make 5 copies of these"
```
{
  "Build": {
    "block_object": {
      "location": {
        "location_type": "SpeakerLook"
      }
    },
    "repeat_for": [1, 1]
  }
}
```

"create copies of every spherical shell"
```
{
  "Build": {
    "block_object": {
      "has_name_": [
        4,
        5
      ],
      "repeat_all": "ALL"
    }
  }
}
```

"help me mine a few holes"
```
{
  "Dig": {
    "repeat_for": [
      3,
      4
    ]
  }
}
```
