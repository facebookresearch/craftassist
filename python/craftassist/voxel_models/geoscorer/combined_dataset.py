"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import numpy as np
import os
import sys

GEOSCORER_DIR = os.path.dirname(os.path.realpath(__file__))
CRAFTASSIST_DIR = os.path.join(GEOSCORER_DIR, "../../")
sys.path.append(CRAFTASSIST_DIR)

import torch
from geoscorer_util import *
from shape_dataset import SegmentContextSeparateShapeData
from inst_seg_dataset import (
    SegmentContextSeparateInstanceData,
    SegmentContextDirectionalInstanceData,
)
from autogen_dataset import SegmentContextSeparateAutogenData


# Returns three tensors: 32x32x32 context, 8x8x8 segment, 1 target
class SegmentContextSeparateData(torch.utils.data.Dataset):
    def __init__(
        self,
        nexamples=100000,
        context_side_length=32,
        seg_side_length=8,
        useid=False,
        for_vis=False,
        ratios={"shape": 1.0},
        extra_params={},
    ):
        self.c_sl = context_side_length
        self.s_sl = seg_side_length
        self.num_examples = nexamples
        self.useid = useid
        self.examples = []
        self.for_vis = for_vis
        self.extra_params = extra_params

        self.ds_names = [k for k, p in ratios.items() if p > 0]
        self.ds_probs = [ratios[name] for name in self.ds_names]
        if sum(self.ds_probs) != 1.0:
            raise Exception("Sum of probs must equal 1.0")
        self.datasets = dict([(name, self._get_dataset(name)) for name in self.ds_names])

    def _get_dataset(self, name):
        if name == "inst_dir":
            min_seg_size = self.extra_params.get("min_seg_size", 6)
            use_saved = self.extra_params.get("use_saved_data", False)
            return SegmentContextDirectionalInstanceData(
                nexamples=self.num_examples,
                min_seg_size=min_seg_size,
                data_preparsed=use_saved,
                save_preparsed=False,
                context_side_length=self.c_sl,
                seg_side_length=self.s_sl,
                useid=self.useid,
            )
        if name == "inst":
            min_seg_size = self.extra_params.get("min_seg_size", 6)
            use_saved = self.extra_params.get("use_saved_data", False)
            return SegmentContextSeparateInstanceData(
                nexamples=self.num_examples,
                min_seg_size=min_seg_size,
                data_preparsed=use_saved,
                save_preparsed=False,
                context_side_length=self.c_sl,
                seg_side_length=self.s_sl,
                useid=self.useid,
                for_vis=self.for_vis,
            )
        if name == "shape":
            return SegmentContextSeparateShapeData(
                nexamples=self.num_examples,
                context_side_length=self.c_sl,
                seg_side_length=self.s_sl,
                useid=self.useid,
                for_vis=self.for_vis,
            )
        if name == "autogen_glue_cubes_dir":
            type_name = self.extra_params.get("type_name", "random")
            fixed_cube_size = self.extra_params.get("fixed_cube_size", None)
            fixed_center = self.extra_params.get("fixed_center", False)
            return SegmentContextSeparateAutogenData(
                nexamples=self.num_examples,
                context_side_length=self.c_sl,
                seg_side_length=self.s_sl,
                useid=self.useid,
                for_vis=self.for_vis,
                type_name=type_name,
                use_direction=True,
                fixed_cube_size=fixed_cube_size,
                fixed_center=fixed_center,
            )
        if name == "autogen_glue_cubes":
            type_name = self.extra_params.get("type_name", "random")
            return SegmentContextSeparateAutogenData(
                nexamples=self.num_examples,
                context_side_length=self.c_sl,
                seg_side_length=self.s_sl,
                useid=self.useid,
                for_vis=self.for_vis,
                type_name=type_name,
                use_direction=False,
            )
        raise Exception("No dataset with name {}".format(name))

    def _get_example(self):
        ds_name = np.random.choice(self.ds_names, p=self.ds_probs)
        return self.datasets[ds_name]._get_example()

    def __getitem__(self, index):
        return self._get_example()

    def __len__(self):
        return self.num_examples


if __name__ == "__main__":
    import os
    import argparse
    import visdom

    VOXEL_MODELS_DIR = os.path.join(GEOSCORER_DIR, "../../")
    sys.path.append(VOXEL_MODELS_DIR)
    import plot_voxels as pv

    parser = argparse.ArgumentParser()
    opts = parser.parse_args()

    vis = visdom.Visdom(server="http://localhost")
    sp = pv.SchematicPlotter(vis)

    dataset = SegmentContextSeparateData(
        nexamples=3,
        for_vis=True,
        useid=False,
        ratios={"autogen_glue_cubes": 1.0},
        extra_params={"min_seg_size": 6},
    )
    for n in range(len(dataset)):
        shape, seg, target = dataset[n]
        sp.drawPlotly(shape)
        sp.drawPlotly(seg)
        target_coord = index_to_coord(target.item(), 32)
        completed_shape = combine_seg_context(seg, shape, target_coord, seg_mult=3)
        sp.drawPlotly(completed_shape)
