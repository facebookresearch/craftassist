"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import json

dataset_config = {
    "inst_dir": [
        {"drop_perc": -1.0, "ground_type": None},
        {"drop_perc": -1.0, "ground_type": "flat"},
    ],
    "shape_dir": [
        {"ground_type": "flat", "max_shift": None},
        {"ground_type": "flat", "max_shift": 6},
        {"ground_type": "hilly", "max_shift": 6},
    ],
    "autogen_glue_cubes_dir": [
        {"fixed_center": True, "ground_type": None},
        {"fixed_center": True, "ground_type": "flat"},
        {"fixed_center": True, "ground_type": "hilly"},
    ],
}
filename = "run_config.json"
with open(filename, "w+") as f:
    json.dump(dataset_config, f)
print("dumped", filename)
