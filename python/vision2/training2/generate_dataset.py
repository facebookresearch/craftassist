#!/usr/bin/python

import numpy as np
import pickle
import random
import sys

sys.path.append("..")
sys.path.append("../..")
from shapes import *
from random_shape_helpers import *
import build_utils as bu
import multiprocessing
from multiprocessing import set_start_method


SHAPE_NAMES = [
    "SPHERE",
    "SPHERICAL_SHELL",
    "RECTANGLE",
    "SQUARE",
    "CUBE",
    "PYRAMID",
    "HOLLOW_CUBE",
]


def convert_insts_to_npy(obj, insts):
    ## regions corresponding to null are 0s
    ret = np.zeros(obj.shape)

    total = 0
    for name, locs_per_class in insts.items():
        for locs_per_inst in locs_per_class:
            total += 1
            for l in locs_per_inst:
                try:
                    ret[l[0], l[1], l[2]] = total
                except:
                    print(name)
                    print(locs_per_inst)
                    assert False

    return ret


def generate_one_sample(i):
    shape_name = random.choice(SHAPE_NAMES)
    options = SHAPE_HELPERS[shape_name]()
    options["bid"] = (np.random.randint(0, 200), 0)
    options["labelme"] = True
    obj, labels, insts = SHAPE_FNS[shape_name](**options)
    obj, _ = bu.blocks_list_to_npy(None, obj, xyz=True)
    obj = obj[:, :, :, 0]  # remove the meta id
    inst_anno = convert_insts_to_npy(obj, insts)
    return obj, inst_anno


def generate_data(n):
    pool = multiprocessing.Pool(40)
    data = pool.map(generate_one_sample, range(n))
    pool.close()
    pool.join()
    return data


if __name__ == "__main__":
    set_start_method("spawn", force=True)

    training_n = 30000
    test_n = 3000

    training_data_ = generate_data(training_n)

    training_data = []
    for td_ in training_data_:
        for td in training_data:
            if np.all(td[0] == td_[0]):
                break
        else:
            training_data.append(td_)
    print(len(training_data))

    ## remove duplicate samples in the training set
    validation_data = []
    for vd in generate_data(test_n):
        for td in training_data:
            if np.all(vd[0] == td[0]):
                break
        else:
            validation_data.append(vd)
    print(len(validation_data))

    with open("./training_data.pkl", "wb") as f:
        pickle.dump(training_data, f)
    with open("./validation_data.pkl", "wb") as f:
        pickle.dump(validation_data, f)
