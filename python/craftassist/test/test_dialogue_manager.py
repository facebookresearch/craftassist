from base_agent.nsp_dialogue_manager import NSPDialogueManager
from base_agent.loco_mc_agent import LocoMCAgent
import os
import unittest
import logging


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


locobot_commands = [
    "go to the gray chair",
    "go to the chair",
    "go forward 0.2 meters",
    "go forward one meter",
    "go left 3 feet",
    "go left 3 meters",
    "go forward 1 feet",
    "go back 1 feet",
    "turn right 90 degrees",
    "turn left 90 degrees",
    "turn right 180 degrees",
    "turn right",
    "look at where I am pointing",
    "wave",
    "follow the chair",
    "push the chair",
    "find Laurens",
    "find the closest red thing",
    "copy this motion",
    "topple the pile of notebooks",
    "bring the cup to Mary",
    "go get me lunch",
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
                "ground_truth_file_path": "ground_truth.txt",
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
        pp = {}
        for x in locobot_commands:
            d = self.agent.dialogue_manager.get_logical_form(x, self.agent.dialogue_manager.model)
            pp[x] = d
        for k, v in pp.items():
            logging.info("\nCommand -> '{}'\nParse -> {}\n".format(k, v))


if __name__ == "__main__":
    unittest.main()
