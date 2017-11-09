#!/usr/bin/python

import os
import argparse
import sys

sys.path.append("..")
from local_voxel_cnn import LocalVoxelCNN
from inst_seg import InstSegModel
from watcher import SubComponentClassifier
from data_loaders import CenterVoxelCubeLoader
from data_loaders import GlobalVoxelLoader
import logging


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inst_seg", action="store_true", help="train a instance segmenation model"
    )
    parser.add_argument("--topk", type=int, default=50, help="How many top labels to use")
    parser.add_argument("--gpu-id", type=int, default=0, help="which gpu to use")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    args = parser.parse_args()

    model_dir = os.path.expanduser("~") + "/minecraft/python/stack_agent_this/models/"

    if args.inst_seg:
        loader_class = GlobalVoxelLoader
        voxel_model_class = InstSegModel
    else:
        loader_class = CenterVoxelCubeLoader
        voxel_model_class = LocalVoxelCNN

    voxel_model_path = model_dir + voxel_model_class.__name__ + ".pth"

    log_file = "logs/" + os.path.basename(voxel_model_path) + ".log"
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="%(asctime)s-[%(filename)s:%(lineno)s - %(funcName)10s()] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    w = SubComponentClassifier(
        voxel_model_class=voxel_model_class,
        voxel_model_opts=dict(gpu_id=args.gpu_id),
        batch_size=args.batch_size,
    )
    w.train_voxel(loader_class(), voxel_model_path, args.topk, lr=1e-4, max_epochs=200)
