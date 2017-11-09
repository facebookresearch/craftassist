import ipdb
import json
import logging
import re
from typing import Tuple, Dict, Optional

import sentry_sdk

import preprocess
from dialogue_manager import DialogueManager
from dialogue_objects import (
    BotCapabilities,
    BotGreet,
    BotVisionDebug,
    DialogueObject,
    GetMemoryHandler,
    Interpreter,
    PutMemoryHandler,
    Say,
    coref_resolve,
    process_spans,
)
from ttad.ttad_model.ttad_model_wrapper import ActionDictBuilder


class TtadModelDialogueManager(DialogueManager):
    def __init__(
        self,
        agent,
        ttad_model_path,
        ttad_embeddings_path,
        ttad_grammar_path,
        no_ground_truth_actions=True,
    ):
        super(TtadModelDialogueManager, self).__init__(agent, None)

        # the following are still scripted and are handled directly from here
        self.botCapabilityQuery = [
            "what can you do",
            "what else can you do",
            "what do you know",
            "tell me what you can do",
            "what things can you do",
            "what are your capabilities",
            "show me what you can do",
            "what are you capable of",
        ]
        self.botGreetings = ["hi", "hello", "hey"]

        logging.info("using ttad_model_path={}".format(ttad_model_path))
        # Instantiate the TTAD model
        if ttad_model_path:
            self.ttad_model = ActionDictBuilder(
                ttad_model_path,
                embeddings_path=ttad_embeddings_path,
                action_tree_path=ttad_grammar_path,
            )
        self.debug_mode = False

        # ground_truth_data is the ground truth action dict from templated
        # generations and will be queried first if checked in.
        if not no_ground_truth_actions:
            with open("ground_truth_data.json") as f:
                self.ground_truth_actions = json.load(f)
        else:
            self.ground_truth_actions = {}

        self.dialogue_object_parameters = {
            "agent": self.agent,
            "memory": self.agent.memory,
            "dialogue_stack": self.dialogue_stack,
        }

    def run_model(self, chat: Tuple[str, str]) -> Optional[DialogueObject]:
        """Process a chat and maybe modify the dialogue stack"""

        if chat[1] == "ipdb":
            ipdb.set_trace()

        if len(self.dialogue_stack) > 0 and self.dialogue_stack[-1].awaiting_response:
            return None

        # chat is a single line command
        speaker, chatstr = chat
        preprocessed_chatstrs = preprocess.preprocess_chat(chatstr)

        # Push appropriate DialogueObjects to stack if incomign chat
        # is one of the scripted ones
        if any([chat in self.botCapabilityQuery for chat in preprocessed_chatstrs]):
            return BotCapabilities(**self.dialogue_object_parameters)
        if any([chat in self.botGreetings for chat in preprocessed_chatstrs]):
            return BotGreet(**self.dialogue_object_parameters)
        if any(["debug_remove" in chat for chat in preprocessed_chatstrs]):
            return BotVisionDebug(**self.dialogue_object_parameters)

        # Assume non-compound command for now
        action_dict = self.ttad(preprocessed_chatstrs[0])
        return self.handle_action_dict(speaker, action_dict)

    def handle_action_dict(self, speaker: str, d: Dict) -> Optional[DialogueObject]:
        """Return the appropriate DialogueObject to handle an action dict "d"

        "d" should have spans resolved by corefs not yet resolved to a specific
        MemoryObject
        """
        coref_resolve(self.agent.memory, d)
        logging.info('ttad post-coref "{}" -> {}'.format(speaker, d))

        if d["dialogue_type"] == "NOOP":
            return Say("I don't know how to answer that.", **self.dialogue_object_parameters)
        elif d["dialogue_type"] == "HUMAN_GIVE_COMMAND":
            return Interpreter(speaker, d, **self.dialogue_object_parameters)
        elif d["dialogue_type"] == "PUT_MEMORY":
            return PutMemoryHandler(speaker, d, **self.dialogue_object_parameters)
        elif d["dialogue_type"] == "GET_MEMORY":
            return GetMemoryHandler(speaker, d, **self.dialogue_object_parameters)
        else:
            raise ValueError("Bad dialogue_type={}".format(d["dialogue_type"]))

    def ttad(self, s: str) -> Dict:
        """Query TTAD model to get the action dict"""

        if s in self.ground_truth_actions:
            d = self.ground_truth_actions[s]
            logging.info('Found gt action for "{}"'.format(s))
        else:
            d = self.ttad_model.parse(["human: " + s])

        # Get the words from indices in spans
        process_spans(d, re.split(r" +", s))
        logging.info('ttad pre-coref "{}" -> {}'.format(s, d))

        # log to sentry
        sentry_sdk.capture_message(json.dumps({"type": "ttad_pre_coref", "in": s, "out": d}))
        return d
