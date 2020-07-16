"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import logging
import numpy as np
import random
from typing import Tuple, Dict, Any, Optional
from word2number.w2n import word_to_num

import sys
import os

BASE_AGENT_ROOT = os.path.join(os.path.dirname(__file__), "../..")
sys.path.append(BASE_AGENT_ROOT)

from base_agent.dialogue_objects import DialogueObject, ConfirmTask, Say, SPEAKERLOOK
from .interpreter_helper import (
    ErrorWithResponse,
    NextDialogueStep,
    get_block_type,
    get_holes,
    get_repeat_num,
    interpret_reference_location,
    interpret_reference_object,
    interpret_schematic,
    interpret_size,
    interpret_stop_condition,
    interpret_facing,
    interpret_point_target,
)
from .modify_helpers import (
    handle_fill,
    handle_rigidmotion,
    handle_scale,
    handle_replace,
    handle_thicken,
)
from .reference_object_helpers import compute_locations
from base_agent.memory_nodes import PlayerNode
from mc_memory_nodes import MobNode
import dance
import tasks


class Interpreter(DialogueObject):
    """This class handles processes incoming chats and modifies the task stack

    Handlers should add/remove/reorder tasks on the stack, but not execute them.
    """

    def __init__(self, speaker: str, action_dict: Dict, **kwargs):
        super().__init__(**kwargs)
        self.speaker = speaker
        self.action_dict = action_dict
        self.provisional: Dict = {}
        self.action_dict_frozen = False
        self.loop_data = None
        self.archived_loop_data = None
        self.default_debug_path = "debug_interpreter.txt"
        self.action_handlers = {
            "MOVE": self.handle_move,
            "BUILD": self.handle_build,
            "DESTROY": self.handle_destroy,
            "DIG": self.handle_dig,
            "STOP": self.handle_stop,
            "RESUME": self.handle_resume,
            "FREEBUILD": self.handle_freebuild,
            "UNDO": self.handle_undo,
            "SPAWN": self.handle_spawn,
            "FILL": self.handle_fill,
            "DANCE": self.handle_dance,
            "MODIFY": self.handle_modify,
            "OTHERACTION": self.handle_otheraction,
        }

    def step(self) -> Tuple[Optional[str], Any]:
        assert self.action_dict["dialogue_type"] == "HUMAN_GIVE_COMMAND"
        try:
            actions = []
            if "action" in self.action_dict:
                actions.append(self.action_dict["action"])
            elif "action_sequence" in self.action_dict:
                actions = self.action_dict["action_sequence"]
                actions.reverse()

            if len(actions) == 0:
                # The action dict is in an unexpected state
                raise ErrorWithResponse(
                    "I thought you wanted me to do something, but now I don't know what"
                )
            for action_def in actions:
                action_type = action_def["action_type"]
                response = self.action_handlers[action_type](self.speaker, action_def)
            return response
        except NextDialogueStep:
            return None, None
        except ErrorWithResponse as err:
            self.finished = True
            return err.chat, None

    def handle_modify(self, speaker, d) -> Tuple[Optional[str], Any]:
        default_ref_d = {"filters": {"location": SPEAKERLOOK}}
        ref_d = d.get("reference_object", default_ref_d)
        # only modify blockobjects...
        objs = interpret_reference_object(
            self, speaker, ref_d, only_physical=True, only_voxels=True
        )
        if len(objs) == 0:
            raise ErrorWithResponse("I don't understand what you want me to modify.")

        m_d = d.get("modify_dict")
        if not m_d:
            raise ErrorWithResponse(
                "I think you want me to modify an object but am not sure what to do"
            )
        for obj in objs:
            if m_d["modify_type"] == "THINNER" or m_d["modify_type"] == "THICKER":
                destroy_task_data, build_task_data = handle_thicken(self, speaker, m_d, obj)
            elif m_d["modify_type"] == "REPLACE":
                destroy_task_data, build_task_data = handle_replace(self, speaker, m_d, obj)
            elif m_d["modify_type"] == "SCALE":
                destroy_task_data, build_task_data = handle_scale(self, speaker, m_d, obj)
            elif m_d["modify_type"] == "RIGIDMOTION":
                destroy_task_data, build_task_data = handle_rigidmotion(self, speaker, m_d, obj)
            elif m_d["modify_type"] == "FILL" or m_d["modify_type"] == "HOLLOW":
                destroy_task_data, build_task_data = handle_fill(self, speaker, m_d, obj)
            else:
                raise ErrorWithResponse(
                    "I think you want me to modify an object but am not sure what to do (parse error)"
                )

            if build_task_data:
                self.append_new_task(tasks.Build, build_task_data)
            if destroy_task_data:
                self.append_new_task(tasks.Destroy, destroy_task_data)

        self.finished = True
        return None, None

    def handle_undo(self, speaker, d) -> Tuple[Optional[str], Any]:
        task_name = d.get("undo_action")
        if task_name:
            task_name = task_name.split("_")[0].strip()
        old_task = self.memory.get_last_finished_root_task(task_name)
        if old_task is None:
            raise ErrorWithResponse("Nothing to be undone ...")
        undo_tasks = [tasks.Undo(self.agent, {"memid": old_task.memid})]

        #        undo_tasks = [
        #            tasks.Undo(self.agent, {"memid": task.memid})
        #            for task in old_task.all_descendent_tasks(include_root=True)
        #        ]
        undo_command = old_task.get_chat().chat_text

        logging.info("Pushing ConfirmTask tasks={}".format(undo_tasks))
        self.dialogue_stack.append_new(
            ConfirmTask,
            'Do you want me to undo the command: "{}" ?'.format(undo_command),
            undo_tasks,
        )
        self.finished = True
        return None, None

    def handle_spawn(self, speaker, d) -> Tuple[Optional[str], Any]:
        spawn_filters = d.get("reference_object", {}).get("filters", {})
        if not spawn_filters:
            raise ErrorWithResponse("I don't understand what you want me to spawn.")

        object_name = spawn_filters["has_name"]
        schematic = self.memory.get_mob_schematic_by_name(object_name)
        if not schematic:
            raise ErrorWithResponse("I don't know how to spawn: %r." % (object_name))

        object_idm = list(schematic.blocks.values())[0]
        mems = interpret_reference_location(self, speaker, SPEAKERLOOK)
        pos, _ = compute_locations(self, speaker, d, mems)
        repeat_times = get_repeat_num(d)
        for i in range(repeat_times):
            task_data = {"object_idm": object_idm, "pos": pos, "action_dict": d}
            self.append_new_task(tasks.Spawn, task_data)
        self.finished = True
        return None, None

    def handle_move(self, speaker, d) -> Tuple[Optional[str], Any]:
        def new_tasks():
            # TODO if we do this better will be able to handle "stay between the x"
            if self.loop_data and hasattr(self.loop_data, "get_pos"):
                pos = self.loop_data.get_pos()
            else:
                location_d = d.get("location", SPEAKERLOOK)
                mems = interpret_reference_location(self, speaker, location_d)
                pos, _ = compute_locations(self, speaker, d, mems)
            # TODO: can this actually happen?
            if pos is None:
                raise ErrorWithResponse("I don't understand where you want me to move.")
            task_data = {"target": pos, "action_dict": d}
            task = tasks.Move(self.agent, task_data)
            return [task]

        if "stop_condition" in d:
            condition = interpret_stop_condition(self, speaker, d["stop_condition"])
            location_d = d.get("location", SPEAKERLOOK)
            mems = interpret_reference_location(self, speaker, location_d)
            if mems is not None and mems:
                self.loop_data = mems[0]
            loop_task_data = {
                "new_tasks_fn": new_tasks,
                "stop_condition": condition,
                "action_dict": d,
            }
            self.append_new_task(tasks.Loop, loop_task_data)
        else:
            for t in new_tasks():
                self.append_new_task(t)

        self.finished = True
        return None, None

    def handle_build(self, speaker, d) -> Tuple[Optional[str], Any]:
        # Get the segment to build
        if "reference_object" in d:
            # handle copy
            repeat = get_repeat_num(d)
            objs = interpret_reference_object(
                self,
                speaker,
                d["reference_object"],
                limit=repeat,
                only_voxels=True,
                loose_speakerlook=True,
            )
            if len(objs) == 0:
                raise ErrorWithResponse("I don't understand what you want me to build")
            tagss = [
                [(p, v) for (_, p, v) in self.memory.get_triples(subj=obj.memid)] for obj in objs
            ]
            interprets = [
                [list(obj.blocks.items()), obj.memid, tags] for (obj, tags) in zip(objs, tagss)
            ]
        else:  # a schematic
            if d.get("repeat") is not None:
                repeat_dict = d
            else:
                repeat_dict = None
            interprets = interpret_schematic(
                self, speaker, d.get("schematic", {}), repeat_dict=repeat_dict
            )

        # Get the locations to build
        location_d = d.get("location", SPEAKERLOOK)
        mems = interpret_reference_location(self, speaker, location_d)
        origin, offsets = compute_locations(
            self, speaker, d, mems, objects=interprets, enable_geoscorer=True
        )
        interprets_with_offsets = [
            (blocks, mem, tags, off) for (blocks, mem, tags), off in zip(interprets, offsets)
        ]

        tasks_todo = []
        for schematic, schematic_memid, tags, offset in interprets_with_offsets:
            og = np.array(origin) + offset
            task_data = {
                "blocks_list": schematic,
                "origin": og,
                "schematic_memid": schematic_memid,
                "schematic_tags": tags,
                "action_dict": d,
            }

            tasks_todo.append(task_data)

        for task_data in reversed(tasks_todo):
            self.append_new_task(tasks.Build, task_data)
        logging.info("Added {} Build tasks to stack".format(len(tasks_todo)))
        self.finished = True
        return None, None

    def handle_freebuild(self, speaker, d) -> Tuple[Optional[str], Any]:
        # This handler handles the action where the agent can complete
        # a human half-built structure using a generative model
        self.dialogue_stack.append_new(Say, "Sorry, I don't know how to do that yet.")
        self.finished = True
        return None, None

    def handle_fill(self, speaker, d) -> Tuple[Optional[str], Any]:
        r = d.get("reference_object")
        self.finished = True
        if not r.get("filters"):
            location_d = SPEAKERLOOK
        else:
            location_d = r["filters"].get("location", SPEAKERLOOK)
        mems = interpret_reference_location(self, speaker, location_d)
        location, _ = compute_locations(self, speaker, d, mems)

        repeat = get_repeat_num(d)
        holes = get_holes(self, speaker, location, limit=repeat)
        if holes is None:
            self.dialogue_stack.append_new(
                Say, "I don't understand what holes you want me to fill."
            )
            return None, None
        for hole in holes:
            _, hole_info = hole
            poss, hole_idm = hole_info
            fill_idm = get_block_type(d["has_block_type"]) if "has_block_type" in d else hole_idm
            task_data = {"action_dict": d, "schematic": poss, "block_idm": fill_idm}
            self.append_new_task(tasks.Fill, task_data)
        if len(holes) > 1:
            self.dialogue_stack.append_new(Say, "Ok. I'll fill up the holes.")
        else:
            self.dialogue_stack.append_new(Say, "Ok. I'll fill that hole up.")
        self.finished = True
        return None, None

    def handle_destroy(self, speaker, d) -> Tuple[Optional[str], Any]:
        default_ref_d = {"filters": {"location": SPEAKERLOOK}}
        ref_d = d.get("reference_object", default_ref_d)
        objs = interpret_reference_object(self, speaker, ref_d, only_destructible=True)
        if len(objs) == 0:
            raise ErrorWithResponse("I don't understand what you want me to destroy.")

        # don't kill mobs
        if all(isinstance(obj, MobNode) for obj in objs):
            raise ErrorWithResponse("I don't kill animals, sorry!")
        if all(isinstance(obj, PlayerNode) for obj in objs):
            raise ErrorWithResponse("I don't kill players, sorry!")
        objs = [obj for obj in objs if not isinstance(obj, MobNode)]
        num_destroy_tasks = 0
        for obj in objs:
            if hasattr(obj, "blocks"):
                schematic = list(obj.blocks.items())
                task_data = {"schematic": schematic, "action_dict": d}
                self.append_new_task(tasks.Destroy, task_data)
                num_destroy_tasks += 1
        logging.info("Added {} Destroy tasks to stack".format(num_destroy_tasks))
        self.finished = True
        return None, None

    # TODO mark in memory it was stopped by command
    def handle_stop(self, speaker, d) -> Tuple[Optional[str], Any]:
        self.finished = True
        if self.loop_data is not None:
            # TODO if we want to be able stop and resume old tasks, will need to store
            self.archived_loop_data = self.loop_data
            self.loop_data = None
        if self.memory.task_stack_pause():
            return "Stopping.  What should I do next?", None
        else:
            return "I am not doing anything", None

    # TODO mark in memory it was resumed by command
    def handle_resume(self, speaker, d) -> Tuple[Optional[str], Any]:
        self.finished = True
        if self.memory.task_stack_resume():
            if self.archived_loop_data is not None:
                # TODO if we want to be able stop and resume old tasks, will need to store
                self.loop_data = self.archived_loop_data
                self.archived_loop_data = None
            return "resuming", None
        else:
            return "nothing to resume", None

    def handle_dig(self, speaker, d) -> Tuple[Optional[str], Any]:
        def new_tasks():
            location_d = d.get("location", SPEAKERLOOK)
            repeat = get_repeat_num(d)
            mems = interpret_reference_location(self, speaker, location_d)
            origin, _ = compute_locations(self, speaker, d, mems)
            attrs = {}
            schematic_d = d["schematic"]
            # set the attributes of the hole to be dug.
            for dim, default in [("depth", 1), ("length", 1), ("width", 1)]:
                key = "has_{}".format(dim)
                if key in schematic_d:
                    attrs[dim] = word_to_num(schematic_d[key])
                elif "has_size" in schematic_d:
                    attrs[dim] = interpret_size(self, schematic_d["has_size"])
                else:
                    attrs[dim] = default

            # add dig tasks in a loop
            z_offset = 0
            tasks_todo = []
            for i in range(repeat):
                og = np.array(origin) + [0, 0, z_offset]  # line them up in +z dir
                t = tasks.Dig(self.agent, {"origin": og, "action_dict": d, **attrs})
                tasks_todo.append(t)
                z_offset += attrs["length"] + 4  # 2-block buffer
            return list(reversed(tasks_todo))

        if "stop_condition" in d:
            condition = interpret_stop_condition(self, speaker, d["stop_condition"])
            self.append_new_task(
                tasks.Loop,
                {"new_tasks_fn": new_tasks, "stop_condition": condition, "action_dict": d},
            )
        else:
            for t in new_tasks():
                self.append_new_task(t)
        self.finished = True
        return None, None

    def handle_otheraction(self, speaker, d) -> Tuple[Optional[str], Any]:
        self.finished = True
        return "I don't know how to do that yet", None

    def handle_dance(self, speaker, d) -> Tuple[Optional[str], Any]:
        def new_tasks():
            repeat = get_repeat_num(d)
            tasks_to_do = []
            # only go around the x has "around"; FIXME allow other kinds of dances
            location_d = d.get("location")
            if location_d is not None:
                rd = location_d.get("relative_direction")
                if rd is not None and (
                    rd == "AROUND" or rd == "CLOCKWISE" or rd == "ANTICLOCKWISE"
                ):
                    ref_obj = None
                    location_reference_object = location_d.get("reference_object")
                    if location_reference_object:
                        objmems = interpret_reference_object(
                            self, speaker, location_reference_object
                        )
                        if len(objmems) == 0:
                            raise ErrorWithResponse("I don't understand where you want me to go.")
                        ref_obj = objmems[0]
                    for i in range(repeat):
                        refmove = dance.RefObjMovement(
                            self.agent,
                            ref_object=ref_obj,
                            relative_direction=location_d["relative_direction"],
                        )
                        t = tasks.Dance(self.agent, {"movement": refmove})
                        tasks_to_do.append(t)
                    return list(reversed(tasks_to_do))

            dance_type = d.get("dance_type", {"dance_type_name": "dance"})
            # FIXME holdover from old dict format
            if type(dance_type) is str:
                dance_type = dance_type = {"dance_type_name": "dance"}
            if dance_type.get("point"):
                target = interpret_point_target(self, speaker, dance_type["point"])
                for i in range(repeat):
                    t = tasks.Point(self.agent, {"target": target})
                    tasks_to_do.append(t)
            # MC bot does not control body turn separate from head
            elif dance_type.get("look_turn") or dance_type.get("body_turn"):
                lt = dance_type.get("look_turn") or dance_type.get("body_turn")
                f = interpret_facing(self, speaker, lt)
                for i in range(repeat):
                    t = tasks.DanceMove(self.agent, f)
                    tasks_to_do.append(t)
            else:
                if location_d is None:
                    dance_location = None
                else:
                    mems = interpret_reference_location(self, speaker, location_d)
                    dance_location, _ = compute_locations(self, speaker, d, mems)
                # TODO use name!
                if dance_type.get("dance_type_span") is not None:
                    dance_name = dance_type["dance_type_span"]
                    if dance_name == "dance":
                        dance_name = "ornamental_dance"
                    dance_memids = self.memory._db_read(
                        "SELECT DISTINCT(Dances.uuid) FROM Dances INNER JOIN Triples on Dances.uuid=Triples.subj WHERE Triples.obj_text=?",
                        dance_name,
                    )
                else:
                    dance_memids = self.memory._db_read(
                        "SELECT DISTINCT(Dances.uuid) FROM Dances INNER JOIN Triples on Dances.uuid=Triples.subj WHERE Triples.obj_text=?",
                        "ornamental_dance",
                    )
                dance_memid = random.choice(dance_memids)[0]
                dance_fn = self.memory.dances[dance_memid]
                for i in range(repeat):
                    dance_obj = dance.Movement(
                        agent=self.agent, move_fn=dance_fn, dance_location=dance_location
                    )
                    t = tasks.Dance(self.agent, {"movement": dance_obj})
                    tasks_to_do.append(t)
            return list(reversed(tasks_to_do))

        if "stop_condition" in d:
            condition = interpret_stop_condition(self, speaker, d["stop_condition"])
            self.append_new_task(
                tasks.Loop,
                {"new_tasks_fn": new_tasks, "stop_condition": condition, "action_dict": d},
            )
        else:
            for t in new_tasks():
                self.append_new_task(t)

        self.finished = True
        return None, None

    def append_new_task(self, cls, data=None):
        # this is badly named, FIXME

        # add a tick to avoid two tasks having same timestamp
        self.memory.add_tick()
        if data is None:
            self.memory.task_stack_push(cls, chat_effect=True)
        else:
            task = cls(self.agent, data)
            self.memory.task_stack_push(task, chat_effect=True)
