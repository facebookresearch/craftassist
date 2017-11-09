#!/usr/bin/python

import argparse
import json
import numpy as np
import torch

from cnn_order import CNN
from .utils import process_break, coord_range


class VoxelCNNGenerator(object):
    def __init__(self, voxel_cnn_gen_dirs):
        # Hyper-parameters
        args = argparse.Namespace()
        args.local_bsize = 7
        args.global_bsize = 21
        args.load = voxel_cnn_gen_dirs
        # args.load = "./generative/model_id17"
        args.block_id = 256
        args.global_block_id = 1
        args.f_dim = 16
        args.history = 3
        args.multi_res = True
        args.embed = False
        args.loss_type = "conditioned"
        args.emb_size = 3

        self.generator = CNN(args)
        self.generator.load_ckpt(args.load)
        self.generator.eval()
        self.hack_flag = True

        if torch.cuda.is_available():
            self.generator = self.generator.cuda()


def build_example():
    # Hack: to see how free build works
    house_data = json.load(open("./generative/sample.json"))
    house, house_data = process_break(house_data)

    # build an object first
    # process break, so we don't need to handle break
    len_ = len(house_data) // 2
    half_house = house_data[:len_]
    xyz_range = coord_range(half_house)
    # xyz_min = np.array(xyz_range[:3]) # get origin as x, y, z min
    xm, ym, zm = xyz_range[:3]  # get origin as x, y, z min
    origin = np.array([0, 64, 0]).astype(np.int)

    blocks_list_free = []
    blocks_list = []

    for item in half_house:
        assert item[-1] == "P"
        x, y, z = item[2]
        x, y, z = x - xm, y - ym, z - zm
        new_block = (x, y, z), tuple(item[3])
        blocks_list.append(new_block)

        x, y, z = item[2]
        x, y, z = x - xm, y - ym + 64, z - zm
        new_block = (x, y, z), tuple(item[3])
        blocks_list_free.append(new_block)

    task_data_free = {"blocks_list": blocks_list_free, "origin": origin}
    task_data = {"blocks_list": blocks_list, "origin": origin}
    return task_data, task_data_free
