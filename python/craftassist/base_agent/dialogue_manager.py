"""
Copyright (c) Facebook, Inc. and its affiliates.

The current control flow of dialogue is:
1.  A chat comes in and Dialogue manager reads it or the bot triggers a
    dialogue because of memory/perception/task state
2.  The dialogue manager puts a DialogueObject on the DialogueStack.
3.  The DialogueStack calls .step() which in turn calls the DialogueObject.step()
    that performs some action as implemented in the step method. The step could
    also possibly interact with the agent's memory. And finally the step() makes
    a call and decides if the DialogueObject is finished.


-   The step() returns a string:  maybe_chat, a dict: maybe_data.
-   The step()'s outputs are read by the manager which can decide to put another
    DialogueObject on the stack.

The maybe_data from the output of the dialogue object's step() can
contain a 'push' key; this overrides the manager's decision on what to push to
the stack.

Control flow for interpreter and clarification:
    The interpreter is also a kind of DialogueObject, and a clarification step is
    the interpreter returning control to the DialogueManager, which pushes a
    ConfirmTask or ConfirmReferenceObject as a DialogueObject onto the DialogueStack.

The manager takes as an input: the agent and the model used for manager.
It creates a DialogueStack.
agent_mem, dialogue_stack, dialogue_object_data, where
dialogue_object_data is for explicit commands to force the manager to
return a specific Dialogue object to put on the stack.
"""
import logging
import os
from typing import Tuple, Optional

from dialogue_stack import DialogueStack
from .dialogue_objects import DialogueObject, Say


class DialogueManager(object):
    def __init__(self, agent, model):
        self.agent = agent
        self.dialogue_stack = DialogueStack(agent, agent.memory)
        self.model = model
        self.safety_words = self.get_safety_words()

    def get_safety_words(self):
        """Read a list of safety words to prevent abuse."""
        with open(os.path.join(os.path.dirname(__file__), "safety.txt")) as f:
            safety_lines = f.readlines()
        safety_words = []
        for l in safety_lines:
            w = l.strip("\n").lower()
            if w != "" and w[0] != "<" and w[0] != "#":
                safety_words.append(w)
        return safety_words

    def check_safety(self, string):
        notsafe = any([string.lower().find(w) > 0 for w in self.safety_words])
        return not notsafe

    # the dialogue manager model should access the task stack and chat history
    # through the agent's memory, adding the most recent chat here as convenience
    # maybe add a get_new_chat to memory and don't pass?
    # chat is a (speaker, str) tuple
    def step(self, chat: Tuple[str, str]):
        # check safety
        if not self.check_safety(chat[1]):
            self.dialogue_stack.append_new(Say, "Please don't be rude.")
            return

        if chat[1]:
            logging.info("Dialogue stack pre-run_model: {}".format(self.dialogue_stack.stack))

            # NOTE: the model is responsible for not putting a new
            # object on the stack if it sees that whatever is on
            # the stack should continue.
            # TODO: Maybe we need a HoldOn dialogue object?
            obj = self.maybe_get_dialogue_obj(chat)
            if obj is not None:
                self.dialogue_stack.append(obj)

        # Always call dialogue_stack.step(), even if chat is empty
        if len(self.dialogue_stack) > 0:
            self.dialogue_stack.step()

    def maybe_get_dialogue_obj(self, chat: Tuple[str, str]) -> Optional[DialogueObject]:
        raise NotImplementedError("Must implement maybe_get_dialogue_object in subclass")
