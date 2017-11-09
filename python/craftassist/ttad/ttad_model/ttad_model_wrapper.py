"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import os

from emnlp_model import *

from typing import Sequence, Dict

THIS_DIR = os.path.dirname(__file__)


class ActionDictBuilder(object):
    def __init__(
        self,
        model_path=THIS_DIR + "/../../models/ttad/ttad.pth",
        embeddings_path=THIS_DIR + "/../../models/ttad/ttad_ft_embeds.pth",
        action_tree_path=THIS_DIR + "/../../models/ttad/dialogue_grammar.json",
        cuda=False,
    ):
        model, loss, w2i, args = load_model(
            model_path=model_path,
            embeddings_path=embeddings_path,
            action_tree_path=action_tree_path,
            use_cuda=cuda,
        )
        self.model = model
        self.loss = loss
        self.w2i = w2i
        self.args = args

    def parse(self, chat_list: Sequence[str]):
        action_dict = predict_tree(self.model, chat_list, self.w2i, self.args)
        self._remove_not_implemented(action_dict)
        return action_dict

    def _remove_not_implemented(self, d: Dict):
        c = d.copy()
        for k, v in c.items():
            if v == "NOT_IMPLEMENTED":
                del d[k]
            elif type(v) == dict:
                self._remove_not_implemented(v)
