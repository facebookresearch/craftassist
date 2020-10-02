"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import numpy as np
import random
import torch
import torch.utils.data
from shape_dataset import SegmentContextShapeData, SegmentContextShapeDirData
from inst_seg_dataset import SegmentContextInstanceData
from autogen_dataset import SegmentContextGlueCubesData


# Returns three tensors: 32x32x32 context, 8x8x8 segment, 1 target
class SegmentContextData(torch.utils.data.Dataset):
    def __init__(
        self,
        nexamples=100000,
        context_side_length=32,
        seg_side_length=8,
        useid=False,
        ratios={"shape": 1.0},
        extra_params={},
        config=None,
    ):
        self.c_sl = context_side_length
        self.s_sl = seg_side_length
        self.num_examples = nexamples
        self.useid = useid
        self.examples = []
        self.extra_params = extra_params

        self.ds_names = [k for k, p in ratios.items() if p > 0]
        self.ds_probs = [ratios[name] for name in self.ds_names]
        if sum(self.ds_probs) != 1.0:
            raise Exception("Sum of probs must equal 1.0")
        if config is None:
            self.datasets = dict(
                [(name, [self._get_dataset(name, extra_params)]) for name in self.ds_names]
            )
        else:
            self.datasets = {}
            for name in self.ds_names:
                if name not in config:
                    continue
                self.datasets[name] = []
                for params in config[name]:
                    self.datasets[name].append(self._get_dataset(name, params))
        print("Datasets")
        for name, dss in self.datasets.items():
            print("   ", name, len(dss))

    def _get_dataset(self, name, extra_params):
        if name == "inst_dir":
            drop_perc = extra_params.get("drop_perc", 0.0)
            ground_type = extra_params.get("ground_type", None)
            return SegmentContextInstanceData(
                use_direction=True,
                nexamples=self.num_examples,
                drop_perc=drop_perc,
                context_side_length=self.c_sl,
                seg_side_length=self.s_sl,
                useid=self.useid,
                ground_type=ground_type,
            )
        if name == "inst":
            drop_perc = extra_params.get("drop_perc", 0.0)
            return SegmentContextInstanceData(
                use_direction=False,
                nexamples=self.num_examples,
                drop_perc=drop_perc,
                context_side_length=self.c_sl,
                seg_side_length=self.s_sl,
                useid=self.useid,
            )
        if name == "shape":
            return SegmentContextShapeData(
                nexamples=self.num_examples,
                context_side_length=self.c_sl,
                seg_side_length=self.s_sl,
                useid=self.useid,
            )
        if name == "shape_dir":
            ground_type = extra_params.get("ground_type", None)
            max_shift = self.extra_params.get("max_shift", 0)
            return SegmentContextShapeDirData(
                nexamples=self.num_examples,
                context_side_length=self.c_sl,
                seg_side_length=self.s_sl,
                useid=self.useid,
                ground_type=ground_type,
                max_shift=max_shift,
            )
        if name == "autogen_glue_cubes_dir":
            type_name = extra_params.get("type_name", "random")
            fixed_cube_size = extra_params.get("fixed_cube_size", None)
            fixed_center = extra_params.get("fixed_center", False)
            ground_type = extra_params.get("ground_type", None)
            return SegmentContextGlueCubesData(
                nexamples=self.num_examples,
                context_side_length=self.c_sl,
                seg_side_length=self.s_sl,
                useid=self.useid,
                type_name=type_name,
                use_direction=True,
                fixed_cube_size=fixed_cube_size,
                fixed_center=fixed_center,
                ground_type=ground_type,
            )
        if name == "autogen_glue_cubes":
            type_name = extra_params.get("type_name", "random")
            return SegmentContextGlueCubesData(
                nexamples=self.num_examples,
                context_side_length=self.c_sl,
                seg_side_length=self.s_sl,
                useid=self.useid,
                type_name=type_name,
                use_direction=False,
            )
        raise Exception("No dataset with name {}".format(name))

    def _get_example(self):
        ds_name = np.random.choice(self.ds_names, p=self.ds_probs)
        dataset = random.choice(self.datasets[ds_name])
        return dataset._get_example()

    def __getitem__(self, index):
        return self._get_example()

    def __len__(self):
        return self.num_examples


if __name__ == "__main__":
    from visualization_utils import GeoscorerDatasetVisualizer

    dataset = SegmentContextData(
        nexamples=3,
        useid=False,
        ratios={"autogen_glue_cubes": 1.0},
        extra_params={"min_seg_size": 6},
    )
    vis = GeoscorerDatasetVisualizer(dataset)
    for n in range(len(dataset)):
        vis.visualize()
