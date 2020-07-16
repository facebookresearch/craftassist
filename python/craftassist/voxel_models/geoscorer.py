"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import logging
import sys
import os
import torch

GEOSCORER_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "geoscorer/")
sys.path.append(GEOSCORER_DIR)

from geoscorer_wrapper import ContextSegmentMergerWrapper
from spatial_utils import shift_sparse_voxel_to_origin, densify


class Geoscorer(object):
    """
    A model class that provides geoscorer functionality.
    This is distinct from the wrapper itself because I see the wrapper
    becoming more specialized as we add more functionality and this object
    possible becoming a process or holding multiple wrappers.
    """

    def __init__(self, merger_model_path=None):
        if merger_model_path is not None:
            logging.info("Geoscorer using  merger_model_path={}".format(merger_model_path))
            self.merger_model = ContextSegmentMergerWrapper(merger_model_path)
        else:
            raise Exception("specify a geoscorer model")
        self.radius = self.merger_model.context_sl // 2
        self.seg_sl = self.merger_model.seg_sl
        self.blacklist = ["BETWEEN", "INSIDE", "AWAY", "NEAR"]

    # Define the circumstances where we can use geoscorer
    def use(self, location_d, repeat_num):
        if repeat_num > 1 or "steps" in location_d:
            return False

        loc_type = location_d.get("location_type", "SPEAKER_LOOK")
        if loc_type == "COORDINATES":
            return False

        rel_dir = location_d.get("relative_direction", None)
        if rel_dir is None or rel_dir in self.blacklist:
            return False
        return True

    def produce_segment_pos_in_context(self, segment, context, brc):
        # Offset puts us right outside of the bottom right corner
        # c_offset = [sum(x) for x in zip(brc, (-1, -1, -1))]
        c_offset = brc
        context_p = self._process_context(context)
        segment_p = self._process_segment(segment)
        bottom_right_coord = self._seg_context_processed_to_coord(segment_p, context_p, c_offset)
        return bottom_right_coord

    def _seg_context_processed_to_coord(self, segment, context, context_off):
        local_coord = self.merger_model.segment_context_to_pos(segment, context)
        global_coord = [sum(x) for x in zip(local_coord, context_off)]
        return global_coord

    def _process_context(self, context):
        c_tensor = torch.from_numpy(context[:, :, :, 0]).long().to(device="cuda")
        return c_tensor

    def _process_segment(self, segment):
        """
        Takes a segment, described as a list of tuples of the form:
            ((x, y, z), (block_id, ?))
        Returns an 8x8x8 block with the segment shifted to the origin its bounds.
        """
        shifted_seg, _ = shift_sparse_voxel_to_origin(segment)

        sl = self.seg_sl
        c = self.seg_sl // 2
        p, _ = densify(shifted_seg, [sl, sl, sl], center=[c, c, c], useid=True)
        s_tensor = torch.from_numpy(p).long().to(device="cuda")
        return s_tensor
