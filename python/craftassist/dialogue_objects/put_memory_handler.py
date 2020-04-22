"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import logging
from typing import Dict, Tuple, Any, Optional

from base_agent.dialogue_objects import DialogueObject
from mc_memory_nodes import ObjectNode, RewardNode
from .interpreter_helper import interpret_reference_object, ErrorWithResponse


class PutMemoryHandler(DialogueObject):
    def __init__(self, speaker_name: str, action_dict: Dict, **kwargs):
        super().__init__(**kwargs)
        self.provisional: Dict = {}
        self.speaker_name = speaker_name
        self.action_dict = action_dict

    def step(self) -> Tuple[Optional[str], Any]:
        r = self._step()
        self.finished = True
        return r

    def _step(self) -> Tuple[Optional[str], Any]:
        assert self.action_dict["dialogue_type"] == "PUT_MEMORY"
        memory_type = self.action_dict["upsert"]["memory_data"]["memory_type"]
        if memory_type == "REWARD":
            return self.handle_reward()
        elif memory_type == "TRIPLE":
            return self.handle_triple()
        else:
            raise NotImplementedError

    def handle_reward(self) -> Tuple[Optional[str], Any]:
        reward_value = self.action_dict["upsert"]["memory_data"]["reward_value"]
        assert reward_value in ("POSITIVE", "NEGATIVE"), self.action_dict
        RewardNode.create(self.memory, reward_value)
        if reward_value == "POSITIVE":
            return "Thank you!", None
        else:
            return "I'll try to do better in the future.", None

    def handle_triple(self) -> Tuple[Optional[str], Any]:
        ref_obj_d = self.action_dict["filters"]["reference_object"]
        r = interpret_reference_object(self, self.speaker_name, ref_obj_d)
        if len(r) == 0:
            raise ErrorWithResponse("I don't know what you're referring to")
        mem = r[0]

        name = "it"
        triples = self.memory.get_triples(subj=mem.memid, pred="has_tag")
        if len(triples) > 0:
            name = triples[0][2].strip("_")

        memory_data = self.action_dict["upsert"]["memory_data"]
        schematic_memid = (
            self.memory.convert_block_object_to_schematic(mem.memid).memid
            if isinstance(mem, ObjectNode)
            else None
        )
        for k, v in memory_data.items():
            if k.startswith("has_"):
                logging.info("Tagging {} {} {}".format(mem.memid, k, v))
                self.memory.add_triple(mem.memid, k, v)
                if schematic_memid:
                    self.memory.add_triple(schematic_memid, k, v)

        point_at_target = mem.get_point_at_target()
        self.agent.send_chat("OK I'm tagging this %r as %r " % (name, v))
        self.agent.point_at(list(point_at_target))

        return "Done!", None
