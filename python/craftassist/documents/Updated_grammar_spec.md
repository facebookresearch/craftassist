This document is WIP and contains different modules / pillars that need to be added to extend the current grammar.
The document uses the concepts of `REFERENCE_OBJECT`, `LOCATION` and `SCHEMATIC` defined in: [Action_dictionary_Spec doc](https://github.com/fairinternal/minecraft/blob/master/python/craftassist/documents/Action_Dictionary_Spec.md)

## Modify action ##
This command represents making a change or modifying a block object in a certain way.

Examples here: https://docs.google.com/document/d/1nLEMRvUO9VNV_HYVRB7c9kDkUl0dACzwjY0h_APDcVU/edit?ts=5e050a45#bookmark=id.o3lwuh3fj1lt

Grammar proposal:
```
{ "dialogue_type": "HUMAN_GIVE_COMMAND",
  "action_sequence" : [
    { "action_type" : 'MODIFY',
			"reference_object" : REFERENCE_OBJECT,
			"location": LOCATION,
			"modifier" : <span of what change needs to be done>
		}]
}
```

If "location" is given, the modify is to move the reference object from its current location to the given location




# Actions for Locobot support #

## Turn ##
This command represents rotating / turning body parts of the bot / robot.
This would cover : Turn / Rotate / Look etc.

Examples here: commands we want the bot to be able to handle

Grammar proposal:
```
{ "dialogue_type": "HUMAN_GIVE_COMMAND",
	"action_sequence" : [
		{ "action_type" : 'TURN',
			"degree" : <span of a degree mentioned in text>,
			"direction" : categorical / span of text representing the direction,
		}]
}
```
## Get ##
This command represents getting or picking up something. This might involve first going to that thing and then picking it up in bot’s hand.

Examples: https://docs.google.com/document/d/1nLEMRvUO9VNV_HYVRB7c9kDkUl0dACzwjY0h_APDcVU/edit?ts=5e050a45#bookmark=id.1fjbolr5xs94

Grammar proposal:
```
{ "dialogue_type": "HUMAN_GIVE_COMMAND",
  "action_sequence" : [
    { "action_type" : 'GET',
			"reference_object": REFERENCE_OBJECT,
		}]
}
```

## Give ##
This command represents giving something, in Minecraft this would mean removing something from the inventory of the bot and adding it to the inventory of the speaker / other player ?
Support from one player to another ?
Support for Show?


Examples: https://docs.google.com/document/d/1nLEMRvUO9VNV_HYVRB7c9kDkUl0dACzwjY0h_APDcVU/edit?ts=5e050a45#bookmark=id.n40i17u2jxaq

Grammar proposal:
```
{ "dialogue_type": "HUMAN_GIVE_COMMAND",
  "action_sequence" : [
    { "action_type" : 'GIVE',
			"reference_object": REFERENCE_OBJECT
		}]
}
```

## Point ##
This command represents the bot pointing at something or some location.

## Bring ##
This command could mean two things:
1. A `MODIFY` for reference objects: “bring the cart from there to where I am standing:
2. A `GET` for Mob or single block: “bring me a diamond”


# Swarm stuff #
Design and define swarm control flow.
This needs further discussion

# Other stuff #
This section includes changes to current grammar which is incremental and needs to be supported in grammar + implementation :
- Add “ACROSS” as a relative direction in location
- Add support for two coordinates in location (from coordinate 1 to coordinate2). Example here: https://docs.google.com/document/d/1nLEMRvUO9VNV_HYVRB7c9kDkUl0dACzwjY0h_APDcVU/edit?ts=5e050a45#bookmark=id.59kednrymynk
- Adding a concept of time in Memory and grammar, so we can enable support for example here: https://docs.google.com/document/d/1nLEMRvUO9VNV_HYVRB7c9kDkUl0dACzwjY0h_APDcVU/edit?ts=5e050a45#bookmark=id.npgh2uui7yln
- Add “dance_type” to Dance to help cover different kinds of movements. Examples here: https://docs.google.com/document/d/1nLEMRvUO9VNV_HYVRB7c9kDkUl0dACzwjY0h_APDcVU/edit?ts=5e050a45#bookmark=id.afthzflclp7


# V/EQA #
This module needs to be fleshed out but some notes about coverage here are:
1. Include queries about attributes: size / color / name / all tags for Mobs and block objects
2. Includes queries of types:
  - Yes/No : “is the house blue ?”
  - What: “what color is the house ?”
3. Queries about bot’s status:
  - Location
  - Current task
  - How far along is it (needs concept of time implemented)
  - Capabilities
4. Extend to all SQL queries:
  - Use cognition labels to get subset
  - Nested queries where input to queries is the output of some function in cognition or current state of the environment
