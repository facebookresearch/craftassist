"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import os
import sys
import visdom
import torch

VOXEL_MODELS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")
sys.path.append(VOXEL_MODELS_DIR)
import plot_voxels as pv
import spatial_utils as su
import training_utils as tu


class GeoscorerDatasetVisualizer:
    def __init__(self, dataset):
        self.vis = visdom.Visdom(server="http://localhost")
        self.sp = pv.SchematicPlotter(self.vis)
        self.dataset = dataset
        self.vis_index = 0
        self.model = None
        self.opts = None

    def set_model(self, model, opts=None):
        self.model = model
        if opts:
            self.opts = opts

    def visualize(self, use_model=False):
        if self.vis_index == len(self.dataset):
            raise Exception("No more examples to visualize in dataset")
        b = self.dataset[self.vis_index]
        if "schematic" in b:
            self.sp.drawPlotly(b["schematic"])
        c_sl = b["context"].size()[0]
        self.vis_index += 1
        self.sp.drawPlotly(b["context"])
        self.sp.drawPlotly(b["seg"])
        target_coord = su.index_to_coord(b["target"].item(), c_sl)
        combined_voxel = su.combine_seg_context(b["seg"], b["context"], target_coord, seg_mult=3)
        self.sp.drawPlotly(combined_voxel)

        if use_model:
            b = {k: t.unsqueeze(0) for k, t in b.items()}
            targets, scores = tu.get_scores_from_datapoint(self.model, b, self.opts)
            max_ind = torch.argmax(scores, dim=1)
            pred_coord = su.index_to_coord(max_ind, c_sl)
            b = {k: t.squeeze(0) for k, t in b.items()}
            predicted_voxel = su.combine_seg_context(
                b["seg"], b["context"], pred_coord, seg_mult=3
            )
            self.sp.drawPlotly(predicted_voxel)
