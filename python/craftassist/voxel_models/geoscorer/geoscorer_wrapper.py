"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import argparse
import models
from geoscorer_util import *


class ContextSegmentMergerWrapper(object):
    """
    Wrapper for the geoscorer
    """

    def __init__(self, models_path):
        if models_path is None:
            raise Exception("Geoscorer wrapper requires a model path")

        self.opts = {}
        tms = models.get_context_segment_trainer_modules(
            opts=self.opts, checkpoint_path=models_path, backup=False, verbose=False
        )
        self.context_net = tms["context_net"]
        self.seg_net = tms["seg_net"]
        self.score_module = tms["score_module"]
        self.context_sl = 32
        self.seg_sl = 8

        self.context_net.eval()
        self.seg_net.eval()
        self.score_module.eval()

    def segment_context_to_pos(self, segment, context):
        # Coords are in Z, X, Y, so put segment into same coords
        segment = segment.permute(1, 2, 0).contiguous()

        c_embed = self.context_net(context.unsqueeze(0))
        s_embed = self.seg_net(segment.unsqueeze(0))
        scores = self.score_module([c_embed, s_embed])
        index = scores[0].flatten().max(0)[1]
        target_coord = index_to_coord(index.item(), self.context_sl)

        # Then take final coord back into X, Y, Z coords
        final_target_coord = (target_coord[2], target_coord[0], target_coord[1])
        return final_target_coord


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_path", type=str, help="path to geoscorer models")
    args = parser.parse_args()

    geoscorer = ContextSegmentMergerWrapper(args.models_path)
