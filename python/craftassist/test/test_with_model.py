"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import os
import unittest

import shapes
from util import euclid_dist
from ttad_model_dialogue_manager import TtadModelDialogueManager
from base_craftassist_test_case import BaseCraftassistTestCase


TTAD_MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models/ttad_bert/model/")
TTAD_BERT_DATA_DIR = os.path.join(os.path.dirname(__file__), "../models/ttad_bert/annotated_data/")


class PutMemoryTestCase(BaseCraftassistTestCase):
    def setUp(self):
        super().setUp()
        self.dialogue_manager = TtadModelDialogueManager(
            self.agent, None, TTAD_MODEL_DIR, TTAD_BERT_DATA_DIR, None, None
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
