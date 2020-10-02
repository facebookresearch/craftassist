"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import unittest
import os
from world import Opt
from base_craftassist_test_case import BaseCraftassistTestCase

GROUND_TRUTH_DATA_DIR = os.path.join(os.path.dirname(__file__), "../datasets/ground_truth/")

"""This class tests common greetings. Tests check whether the command executed successfully 
without world state changes; for correctness inspect chat dialogues in logging.
"""


class GreetingTest(BaseCraftassistTestCase):
    def setUp(self):
        opts = Opt()
        opts.ground_truth_data_dir = GROUND_TRUTH_DATA_DIR
        opts.no_ground_truth = False
        opts.nsp_model_dir = None
        opts.nsp_data_dir = None
        opts.nsp_embedding_path = None
        opts.model_base_path = None
        opts.QA_nsp_model_path = None
        opts.web_app = False
        super().setUp(agent_opts=opts)

    def test_hello(self):
        self.add_incoming_chat("hello", self.speaker)
        changes = self.flush()
        self.assertFalse(changes)

    def test_goodbye(self):
        self.add_incoming_chat("goodbye", self.speaker)
        changes = self.flush()
        self.assertFalse(changes)


if __name__ == "__main__":
    unittest.main()
