"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import os
import unittest

import shapes
from util import euclid_dist
from ttad_model_dialogue_manager import TtadModelDialogueManager
from base_craftassist_test_case import BaseCraftassistTestCase


TTAD_MODELS_DIR = os.path.dirname(__file__) + "/../models/ttad"


class PutMemoryTestCase(BaseCraftassistTestCase):
    def setUp(self):
        super().setUp()
        self.dialogue_manager = TtadModelDialogueManager(
            self.agent,
            TTAD_MODELS_DIR + "/ttad.pth",
            TTAD_MODELS_DIR + "/ttad_ft_embeds.pth",
            TTAD_MODELS_DIR + "/dialogue_grammar.json",
        )

        self.cube_right = self.add_object(shapes.cube(bid=(42, 0)), (9, 63, 4))
        self.cube_left = self.add_object(shapes.cube(), (9, 63, 10))
        self.set_looking_at(list(self.cube_right.blocks.keys())[0])

    def test_come_here(self):
        chat = "come here"
        self.add_incoming_chat(chat)
        self.dialogue_manager.step((self.speaker, chat))
        self.flush()

        self.assertLessEqual(euclid_dist(self.agent.pos, self.get_speaker_pos()), 1)

    def test_stop(self):
        chat = "stop"
        self.add_incoming_chat(chat)
        self.dialogue_manager.step((self.speaker, chat))
        self.flush()


if __name__ == "__main__":
    unittest.main()
