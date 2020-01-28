"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import numpy as np
import unittest
from unittest.mock import Mock

from build_utils import to_relative_pos
from dialogue_objects import AwaitResponse
from fake_agent import FakeAgent
from memory import AgentMemory
from memory_nodes import ObjectNode
from ttad_model_dialogue_manager import TtadModelDialogueManager
from typing import List, Sequence, Dict
from util import XYZ, Block, IDM, pos_to_np
from utils import Pos


class BaseCraftassistTestCase(unittest.TestCase):
    def setUp(self):
        self.memory = AgentMemory(load_minecraft_specs=False)  # don't load specs, it's slow
        self.agent = FakeAgent(self.memory)
        self.dialogue_manager = TtadModelDialogueManager(
            self.agent, None, None, None, None, None, no_ground_truth_actions=True
        )

        # More helpful error message to encourage test writers to use self.set_looking_at()
        self.agent.get_player_line_of_sight = Mock(
            side_effect=NotImplementedError(
                "Cannot call into C++ function in this unit test. "
                + "Call self.set_looking_at() to set the return value"
            )
        )

        # Add a speaker at position (5, 63, 5) looking in the +x direction
        self.memory.update(self.agent)
        self.speaker = list(self.memory.other_players.values())[0].name

        # Combinable actions to be used in test cases
        self.possible_actions = {
            "destroy_speaker_look": {
                "action_type": "DESTROY",
                "reference_object": {"location": {"location_type": "SPEAKER_LOOK"}},
            },
            "copy_speaker_look_to_agent_pos": {
                "action_type": "BUILD",
                "reference_object": {"location": {"location_type": "SPEAKER_LOOK"}},
                "location": {"location_type": "AGENT_POS"},
            },
            "build_small_sphere": {
                "action_type": "BUILD",
                "schematic": {"has_name": "sphere", "has_size": "small"},
            },
            "build_1x1x1_cube": {
                "action_type": "BUILD",
                "schematic": {"has_name": "cube", "has_size": "1 x 1 x 1"},
            },
            "move_speaker_pos": {
                "action_type": "MOVE",
                "location": {"location_type": "SPEAKER_POS"},
            },
            "build_diamond": {"action_type": "BUILD", "schematic": {"has_name": "diamond"}},
            "build_gold_cube": {
                "action_type": "BUILD",
                "schematic": {"has_block_type": "gold", "has_name": "cube"},
            },
            "fill_all_holes_speaker_look": {
                "action_type": "FILL",
                "location": {"location_type": "SPEAKER_LOOK"},
                "repeat": {"repeat_key": "ALL"},
            },
            "go_to_tree": {
                "action_type": "MOVE",
                "location": {
                    "location_type": "REFERENCE_OBJECT",
                    "reference_object": {"has_name": "tree"},
                },
            },
            "build_square_height_1": {
                "action_type": "BUILD",
                "schematic": {"has_name": "square", "has_height": "1"},
            },
            "stop": {"action_type": "STOP"},
            "fill_speaker_look": {
                "action_type": "FILL",
                "location": {"location_type": "SPEAKER_LOOK"},
            },
            "fill_speaker_look_gold": {
                "action_type": "FILL",
                "has_block_type": "gold",
                "location": {"location_type": "SPEAKER_LOOK"},
            },
        }

    def handle_action_dict(
        self, d, chatstr: str = "", answer: str = None, stop_on_chat=False, max_steps=10000
    ) -> Dict[XYZ, IDM]:
        """Handle an action dict and call self.flush()

        If "answer" is specified and a question is asked by the agent, respond
        with this string.

        If "stop_on_chat" is specified, stop iterating if the agent says anything
        """
        self.add_incoming_chat("TEST {}".format(d))
        # FIXME!  sending empty chat to coref resolve unless passed in!!
        obj = self.dialogue_manager.handle_action_dict(self.speaker, d, chatstr)
        if obj is not None:
            self.dialogue_manager.dialogue_stack.append(obj)
        changes = self.flush(max_steps, stop_on_chat=stop_on_chat)
        if len(self.dialogue_manager.dialogue_stack) != 0 and answer is not None:
            self.add_incoming_chat(answer)
            changes.update(self.flush(max_steps, stop_on_chat=stop_on_chat))
        return changes

    def flush(self, max_steps=10000, stop_on_chat=False) -> Dict[XYZ, IDM]:
        """Update memory and step the dialogue and task stacks until they are empty

        If "stop_on_chat" is specified, stop iterating if the agent says anything

        Return the set of blocks that were changed.
        """
        if stop_on_chat:
            self.agent.clear_outgoing_chats()

        world_before = self.agent._world.copy()

        for _ in range(max_steps):
            if (
                len(self.dialogue_manager.dialogue_stack) == 0
                and not self.memory.task_stack_peek()
            ):
                break
            self.memory.update(self.agent)
            self.dialogue_manager.dialogue_stack.step()
            self.agent.task_step()
            if (
                isinstance(self.dialogue_manager.dialogue_stack.peek(), AwaitResponse)
                and not self.dialogue_manager.dialogue_stack.peek().finished
            ) or (stop_on_chat and self.agent.get_last_outgoing_chat()):
                break
        self.memory.update(self.agent)

        # get changes
        world_after = self.agent._world.copy()
        changes = dict(set(world_after.items()) - set(world_before.items()))
        changes.update({k: (0, 0) for k in set(world_before.keys()) - set(world_after.keys())})
        return changes

    def set_looking_at(self, xyz: XYZ):
        """Set the return value for C++ call to get_player_line_of_sight"""
        self.agent.get_player_line_of_sight = Mock(return_value=Pos(*xyz))

    def set_blocks(self, xyzbms: List[Block], origin: XYZ = (0, 0, 0)):
        """Change the state of the world, block by block"""
        for xyz, idm in xyzbms:
            abs_xyz = tuple(np.array(xyz) + origin)
            self.memory.on_block_changed(abs_xyz, idm)
            self.agent._world[abs_xyz] = idm

    def add_object(self, xyzbms: List[Block], origin: XYZ = (0, 0, 0)) -> ObjectNode:
        """Add an object to memory as if it was placed block by block

        Args:
        - xyzbms: a list of relative (xyz, idm)
        - origin: (x, y, z) of the corner

        Returns an ObjectNode
        """
        self.set_blocks(xyzbms, origin)
        abs_xyz = tuple(np.array(xyzbms[0][0]) + origin)
        memid = self.memory.get_block_object_ids_by_xyz(abs_xyz)[0]
        return self.memory.get_object_by_id(memid)

    def get_blocks(self, xyzs: Sequence[XYZ]) -> Dict[XYZ, IDM]:
        """Return the ground truth block state"""
        d = {}
        for (x, y, z) in xyzs:
            B = self.agent.get_blocks(x, x, y, y, z, z)
            d[(x, y, z)] = tuple(B[0, 0, 0, :])
        return d

    def add_incoming_chat(self, chat: str):
        """Add a chat to memory as if it was just spoken by SPEAKER"""
        self.memory.add_chat(self.memory.get_player_by_name(self.speaker).memid, chat)

    def assert_schematics_equal(self, a, b):
        """Check equality between two list[(xyz, idm)] schematics

        N.B. this compares the shapes and idms, but ignores absolute position offsets.
        """
        a, _ = to_relative_pos(a)
        b, _ = to_relative_pos(b)
        self.assertEqual(set(a), set(b))

    def last_outgoing_chat(self) -> str:
        return self.agent.get_last_outgoing_chat()

    def get_speaker_pos(self) -> XYZ:
        return tuple(pos_to_np(self.memory.get_player_struct_by_name(self.speaker).pos))
