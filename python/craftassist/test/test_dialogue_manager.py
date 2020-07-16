from base_agent.nsp_dialogue_manager import NSPDialogueManager
from base_agent.loco_mc_agent import LocoMCAgent
import os
import unittest
import logging
from all_test_commands import *


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class FakeAgent(LocoMCAgent):
    def __init__(self, opts):
        super(FakeAgent, self).__init__(opts)
        self.opts = opts

    def init_memory(self):
        self.memory = "memory"

    def init_physical_interfaces(self):
        pass

    def init_perception(self):
        pass

    def init_controller(self):
        dialogue_object_classes = {}
        self.dialogue_manager = NSPDialogueManager(self, dialogue_object_classes, self.opts)


# NOTE: The following commands in locobot_commands can't be supported
# right away but we'll attempt them in the next round:
# "push the chair",
# "find the closest red thing",
# "copy this motion",
# "topple the pile of notebooks",
locobot_commands = list(GROUND_TRUTH_PARSES) + [
    "push the chair",
    "find the closest red thing",
    "copy this motion",
    "topple the pile of notebooks",
]


class TestDialogueManager(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestDialogueManager, self).__init__(*args, **kwargs)
        opts = AttributeDict(
            {
                "QA_nsp_model_path": "models/ttad/ttad.pth",
                "nsp_embeddings_path": "models/ttad/ttad_ft_embeds.pth",
                "nsp_grammar_path": "models/ttad/dialogue_grammar.json",
                "nsp_data_dir": "models/ttad_bert_updated/annotated_data/",
                "nsp_model_dir": "models/ttad_bert_updated/model/",
                "ground_truth_data_dir": "models/ttad_bert_updated/ground_truth/",
                "web_app": False,
            }
        )

        def fix_path(opts):
            base_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "craftassist"
            )
            for optname, optval in opts.items():
                if "path" in optname or "dir" in optname:
                    if optval:
                        opts[optname] = os.path.join(base_path, optval)

        fix_path(opts)
        self.agent = FakeAgent(opts)

    def test_parses(self):
        logging.info(
            "Printing semantic parsing for {} locobot commands".format(len(locobot_commands))
        )

        for command in locobot_commands:
            ground_truth_parse = GROUND_TRUTH_PARSES.get(command, None)
            model_prediction = self.agent.dialogue_manager.get_logical_form(
                command, self.agent.dialogue_manager.model
            )

            logging.info(
                "\nCommand -> '{}' \nGround truth -> {} \nParse -> {}\n".format(
                    command, ground_truth_parse, model_prediction
                )
            )


if __name__ == "__main__":
    unittest.main()
