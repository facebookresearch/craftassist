import argparse
from cnn_order import CNN

import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-local_bsize", default=7, type=int, help="local box dimension")
    parser.add_argument("-global_bsize", default=21, type=int, help="global box dimension")
    parser.add_argument("-load", type=str, default="model_id17", help="checkpoint to load from")
    parser.add_argument("-block_id", type=int, default=256, help="how many types of block_id")
    parser.add_argument(
        "-global_block_id", type=int, default=1, help="how many types of global_block_id"
    )
    parser.add_argument("-f_dim", type=int, default=16, help="feature dim")
    parser.add_argument("-history", type=int, default=3, help="history history to use")
    parser.add_argument("-multi_res", type=bool, default=True, help="multi-resolution")
    parser.add_argument("-embed", type=bool, default=False, help="embed before cnn")
    parser.add_argument(
        "-loss_type", type=str, default="conditioned", help="loss type: [regular, regression]"
    )

    parser.add_argument("-emb_size", type=int, default=3, help="block embedding size")
    args = parser.parse_args()

    model = CNN(args)

    if args.load != "":
        model.load_state_dict(torch.load(args.load))
