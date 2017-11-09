import sys
import os
import random
import torch
import numpy as np
import argparse
import datetime
import logging

# constants
RANK_NORM_FACTOR = 10
DEBUG = False


class Config:
    def __init__(self):
        parser = argparse.ArgumentParser()
        # general
        parser.add_argument(
            "-seed", "--seed", type=int, help="Torch Random Seed", required=False, default=1
        )
        parser.add_argument("-mode", "--mode", help="Mode", required=False, default="train")
        parser.add_argument(
            "-cuda", "--cuda", type=bool, help="Use CUDA or not", required=False, default=True
        )
        parser.add_argument(
            "-gpu_id", "--gpu_id", type=int, help="Gpu device id", required=False, default=0
        )
        parser.add_argument(
            "-vis_mode",
            "--vis_mode",
            type=str,
            help="Visual evaluation mode: concatenate keywords [local, global, final]",
            required=False,
            default="none",
        )

        # paths
        parser.add_argument(
            "-data_dir",
            "--data_dir",
            type=str,
            help="Data Folder",
            required=False,
            default="../../../../house_segment/order_data/real/",
        )
        parser.add_argument(
            "-log_dir",
            "--log_dir",
            type=str,
            help="Logging Directory",
            required=False,
            default="../log_dir/",
        )
        parser.add_argument(
            "-model_dir",
            "--model_dir",
            type=str,
            help="Model Directory",
            required=False,
            default="../model_dir/",
        )
        parser.add_argument(
            "-load_model",
            "--load_model",
            type=str,
            help="Load Model checkpoint",
            required=False,
            default="",
        )

        # optimization
        parser.add_argument(
            "-batch_size", "--batch_size", type=int, help="Batch Size", required=False, default=32
        )
        parser.add_argument(
            "-epochs", "--epochs", type=int, help="Epochs", required=False, default=50
        )
        parser.add_argument(
            "-lr", "--lr", type=float, help="Learning Rate", required=False, default=0.001
        )
        parser.add_argument(
            "-wd", "--wd", type=float, help="Weight Decay", required=False, default=0.01
        )

        # model
        parser.add_argument(
            "-use_schematic",
            "--use_schematic",
            type=bool,
            help="Whether to use Schematic",
            required=False,
            default=False,
        )
        parser.add_argument(
            "-k_neg",
            "--k_neg",
            type=int,
            help="# of negative examples",
            required=False,
            default=10,
        )
        parser.add_argument(
            "-boxsize", "--boxsize", type=int, help="Box Size", required=False, default=15
        )
        parser.add_argument(
            "-delta_constant",
            "--delta_constant",
            type=float,
            help="Delta Constant",
            required=False,
            default=0.3,
        )
        parser.add_argument(
            "-h_dim", "--h_dim", type=int, help="Hidden Dimension", required=False, default=8
        )
        parser.add_argument(
            "-dropout_rate",
            "--dropout_rate",
            type=float,
            help="Dropout Rate",
            required=False,
            default=0.8,
        )

        self.args = parser.parse_args()

        # set device
        if torch.cuda.is_available() and self.args.cuda:
            self.use_cuda = True
            torch.cuda.manual_seed(self.args.seed)
            torch.cuda.set_device(self.args.gpu_id)
        else:
            self.use_cuda = False
        self.kwargs = {"num_workers": 4, "pin_memory": True} if self.use_cuda else {}

        # set random seed
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        # set data files
        self.train_source = os.path.join(self.args.data_dir, "train.data.pkl")
        self.val_source = os.path.join(self.args.data_dir, "val.data.pkl")
        self.test_source = os.path.join(self.args.data_dir, "test.data.pkl")

        # set model identifier
        now = datetime.datetime.now()
        self.model_identifier = "model-default-%s_%s_%s_%s_%s" % (
            now.year,
            now.month,
            now.day,
            now.hour,
            now.minute,
        )

        # make directories
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
        self.args.model_dir = os.path.join(self.args.model_dir, self.model_identifier)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        self.args.visual_dir = os.path.join(self.args.model_dir, "visualize")
        self.args.model_dir = os.path.join(self.args.model_dir, "saved_checkpoints")
        self.args.visual_counter = 0
        if not os.path.exists(self.args.visual_dir):
            os.makedirs(self.args.visual_dir)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)

        # set logging
        self.log_file = "%s/%s.log" % (self.args.log_dir, self.model_identifier)
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.DEBUG)
        self.fh = logging.FileHandler(self.log_file)
        self.fh.setLevel(logging.DEBUG)
        self.ch = logging.StreamHandler(sys.stdout)
        self.ch.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S")
        self.fh.setFormatter(formatter)
        self.ch.setFormatter(formatter)
        self.log.addHandler(self.fh)
        self.log.addHandler(self.ch)

        self.log.info("log to %s" % self.log_file)
        self.log.info("Config: %s" % self.__dict__)

        # TODO(demiguo): load label dict
        label_dict = {}
        self.vsize = 3 if not self.args.use_schematic else len(label_dict + 1)
