#!/usr/bin/python

import os
import argparse
import sys

sys.path.append("..")
from local_voxel_cnn import LocalVoxelCNN
from watcher import SubComponentClassifier
from data_loaders import SoftmaxLoader, CRFDataLoader
import logging


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--voxel", action="store_true", help="train a voxel model")
    parser.add_argument("--crf", action="store_true", help="train a CRF model")
    parser.add_argument("--top-k", type=int, default=20, help="How many label classes to train")
    parser.add_argument("--log", type=str, default="", help="Write to a log file")
    parser.add_argument("--voxel-group", action="store_true", help="Whether use voxel groups")
    args = parser.parse_args()

    model_dir = os.path.expanduser("~") + "/minecraft/python/craftassist/models/"

    voxel_model_path = (
        model_dir
        + "voxel-"
        + ("g-" if args.voxel_group else "")
        + LocalVoxelCNN.__name__
        + ("-t%d.pth" % args.top_k)
    )

    if args.log != "":
        log_file = args.log
    else:
        log_file = "logs/" + os.path.basename(voxel_model_path) + ".log"

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    if args.voxel:
        loader = SoftmaxLoader(["training0", "training1"], pred_topk=3)
        w = SubComponentClassifier(
            voxel_model_class=LocalVoxelCNN,
            voxel_model_opts=dict(voxel_group=args.voxel_group),
            batch_size=64,
        )
        w.train_voxel(loader, voxel_model_path, top_k=args.top_k, lr=1e-3, max_epochs=300)

    if args.crf:
        crf_loader = CRFDataLoader(["training0", "training1"])
        crf_model_path = model_dir + "/crf.npy"
        w = SubComponentClassifier(voxel_model_path=voxel_model_path, batch_size=64)
        w.train_crf(
            crf_loader,
            crf_model_path,
            pairwise_weight_range=[8, 10, 12],
            sxyz_range=[3.0, 5.0, 8.0],
            sembed_range=[1.8, 2.0, 2.2, 2.5],
        )
