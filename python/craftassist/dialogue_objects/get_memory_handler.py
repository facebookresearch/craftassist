"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

from typing import Dict, Tuple, Any, Optional, Sequence

from base_agent.dialogue_objects import DialogueObject
from .interpreter_helper import interpret_reference_object, ErrorWithResponse
from base_agent.memory_nodes import MemoryNode, ReferenceObjectNode
from string_lists import ACTION_ING_MAPPING
from tasks import Build
from ttad.generation_dialogues.generate_utils import prepend_a_an


######FIXME TEMPORARY:
from base_agent import post_process_logical_form


class GetMemoryHandler(DialogueObject):
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
        assert self.action_dict["dialogue_type"] == "GET_MEMORY"

        filter_type = self.action_dict["filters"]["type"]
        if filter_type == "ACTION":
            return self.handle_action()
        elif filter_type == "AGENT":
            return self.handle_agent()
        elif filter_type == "REFERENCE_OBJECT":
            return self.handle_reference_object()
        else:
            raise ValueError("Unknown filter_type={}".format(filter_type))

    def handle_reference_object(self, voxels_only=False) -> Tuple[Optional[str], Any]:
        ####FIXME TEMPORARY!!!!
        r = post_process_logical_form.fix_reference_object_with_filters(
            self.action_dict["filters"]
        )
        objs = interpret_reference_object(
            self, self.speaker_name, r["reference_object"], not_location=True
        )
        return self.do_answer(objs)

    def handle_action(self) -> Tuple[Optional[str], Any]:
        # get current action
        target_action_type = self.action_dict["filters"].get("target_action_type")
        if target_action_type:
            task = self.memory.task_stack_find_lowest_instance(target_action_type)
        else:
            task = self.memory.task_stack_peek()
            if task is not None:
                task = task.get_root_task()
        if task is None:
            return "I am not doing anything right now", None

        # get answer
        return self.do_answer([task])

    def handle_agent(self) -> Tuple[Optional[str], Any]:
        # location is currently the only expected answer_type
        location = tuple(self.agent.pos)
        return "I am at {}".format(location), None

    def do_answer(self, mems: Sequence[MemoryNode]) -> Tuple[Optional[str], Any]:
        answer_type = self.action_dict["answer_type"]
        if answer_type == "TAG":
            return self.handle_answer_type_tag(mems)
        elif answer_type == "EXISTS":
            return self.handle_answer_type_exists(mems)
        else:
            raise ValueError("Bad answer_type={}".format(answer_type))

    def handle_answer_type_tag(self, mems: Sequence[MemoryNode]) -> Tuple[Optional[str], Any]:
        if len(mems) == 0:
            raise ErrorWithResponse("I don't know what you're referring to")
        mem = mems[0]
        tag_name = self.action_dict["tag_name"]
        if tag_name.startswith("has_"):
            triples = self.memory.get_triples(subj=mem.memid, pred_text=tag_name)
            # TODO backoff to other memories, search etc.
            if len(triples) == 0:
                # first backoff to tags
                triples = self.memory.get_triples(subj=mem.memid, pred_text="has_tag")
                if len(triples) == 0:
                    return "I don't know", None
                else:
                    tag_name = "has_tag"
            all_tags = [t[2] for t in triples if t[2][0] != "_"]
            _, _, val = triples[0]
            if tag_name == "has_name":
                if "_in_progress" in self.memory.get_tags_by_memid(mem.memid):
                    return "It will be a %r" % (val), None
                else:
                    return "It is a %r" % (val), None
            elif tag_name == "has_tag":
                return "That has tags " + " ".join(all_tags), None
            else:
                return "It is %r" % (val), None
        elif tag_name == "action_name":
            assert hasattr(mem, "task")
            return "I am {}".format(ACTION_ING_MAPPING[mem.action_name.lower()]), None
        elif tag_name == "action_reference_object_name":
            assert hasattr(mems[0], "task"), mems[0]
            assert isinstance(mems[0].task, Build), mems[0].task
            for pred, val in mems[0].task.schematic_tags:
                if pred == "has_name":
                    return "I am building " + prepend_a_an(val), None
            return "I am building something that is {}".format(val), None
        elif tag_name == "move_target":
            assert mem.action_name == "Move", mem
            target = tuple(mem.task.target)
            return "I am going to {}".format(target), None
        elif tag_name == "location":
            if isinstance(mems[0], ReferenceObjectNode):
                return str(mems[0].get_pos()), None
            else:
                raise TypeError("Can't get location of {} {}".format(mems[0], mems[0].memid))
        else:
            raise ErrorWithResponse("I don't understand what you're asking")

    def handle_answer_type_exists(self, mems: Sequence[MemoryNode]) -> Tuple[Optional[str], Any]:
        # we check progeny data bc if it exists, there was a confirmation,
        # and the interpret reference object failed to find the object
        # so it does not have the proper tag.  this is an unused opportunity to learn...
        # also note if the answer is going to be no, bot will always ask.  maybe should fix this.
        if len(mems) > 0 and len(self.progeny_data) == 0:
            return "Yes", None
        else:
            return "No", None
