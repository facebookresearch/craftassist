"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import os
import unittest

import shapes
from util import euclid_dist
from base_craftassist_test_case import BaseCraftassistTestCase


class Opt:
    pass


TTAD_MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models/ttad_bert_updated/model/")
TTAD_BERT_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "../models/ttad_bert_updated/annotated_data/"
)


class PutMemoryTestCase(BaseCraftassistTestCase):
    def setUp(self):
        opts = Opt()
        opts.nsp_model_dir = TTAD_MODEL_DIR
        opts.nsp_data_dir = TTAD_BERT_DATA_DIR
        opts.nsp_embedding_path = None
        opts.model_base_path = None
        opts.QA_nsp_model_path = None
        opts.ground_truth_data_dir = ""
        opts.web_app = False
        super().setUp(agent_opts=opts)

        self.cube_right = self.add_object(shapes.cube(bid=(42, 0)), (9, 63, 4))
        self.cube_left = self.add_object(shapes.cube(), (9, 63, 10))
        self.set_looking_at(list(self.cube_right.blocks.keys())[0])

    def test_come_here(self):
        chat = "come here"
        self.add_incoming_chat(chat, self.speaker)
        self.flush()

        self.assertLessEqual(euclid_dist(self.agent.pos, self.get_speaker_pos()), 1)

    def test_stop(self):
        chat = "stop"
        self.add_incoming_chat(chat, self.speaker)
        self.flush()


if __name__ == "__main__":
    unittest.main()
