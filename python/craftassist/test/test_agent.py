"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import unittest

from craftassist_agent import CraftAssistAgent


class Opt:
    pass


class BaseAgentTest(unittest.TestCase):
    def test_init_agent(self):
        opts = Opt()
        opts.no_default_behavior = False
        opts.semseg_model_path = None
        opts.geoscorer_model_path = None
        opts.nsp_model_dir = None
        opts.nsp_data_dir = None
        opts.nsp_embedding_path = None
        opts.model_base_path = None
        opts.QA_nsp_model_path = None
        opts.ground_truth_data_dir = ""
        opts.semseg_model_path = ""
        opts.geoscorer_model_path = ""
        opts.web_app = False
        # test does not instantiate cpp client
        opts.port = -1
        opts.no_default_behavior = False
        CraftAssistAgent(opts)


if __name__ == "__main__":
    unittest.main()
