import torch
import argparse
import random


class Tree2TextDataset(torch.utils.data.Dataset):
    """Dataset class extending pytorch Dataset definition.
    """

    def __init__(self, prog, chat):
        """Initialize paired data, tokenizer, dictionaries.
        """
        assert len(prog) == len(chat)
        self.prog = prog
        self.chat = chat

    def __len__(self):
        """Returns total number of samples.
        """
        return len(self.prog)

    def __getitem__(self, index):
        """Generates one sample of data.
        """
        X = self.prog[index]
        Y = self.chat[index]
        return (X, Y)


def load_paired_dataset(data_dir, prefix):
    chat_data = []
    prog_data = []
    with open(data_dir + "{}.chat".format(prefix)) as fd:
        chat_data = fd.read().splitlines()
    with open(data_dir + "{}.prog".format(prefix)) as fd:
        prog_data = fd.read().splitlines()
    return (prog_data, chat_data)


def collate(tree, text, tree_tokenizer, text_tokenizer):
    """Pad and tensorize data.
    """
    # Longest tree
    max_x_len = max([len(x) for x in tree])
    x_mask = [[1] * len(x) + [0] * (max_x_len - len(x)) for x in tree]
    batch_x_with_pad = [x + [tree_tokenizer.pad_token_id] * (max_x_len - len(x)) for x in tree]
    # Longest text command
    max_y_len = max([len(y) for y in text])
    y_mask = [[1] * len(y) + [0] * (max_y_len - len(y)) for y in text]
    batch_y_with_pad = [y + [text_tokenizer.pad_token_id] * (max_y_len - len(y)) for y in text]
    # Convert padded data to tensors
    x = batch_x_with_pad
    x_mask = x_mask
    y = batch_y_with_pad
    y_mask = y_mask
    return (x, x_mask, y, y_mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="/checkpoint/rebeccaqian/datasets/fairseq/07_25/",
        type=str,
        help="train/valid/test data",
    )
    parser.add_argument(
        "--prefix", default="train", type=str, help="partition to load eg. train, valid"
    )
    # Load dataset and print some meta stats.
    args = parser.parse_args()
    train_prog_data, train_chat_data = load_paired_dataset(args.data_dir, args.prefix)
    train_dataset = Tree2TextDataset(train_prog_data, train_chat_data)
    length = train_dataset.__len__()
    print("Size of data: {}".format(length))
    # Randomly generate some samples.
    for i in range(5):
        rand = random.uniform(0, 1)
        idx = round(rand * length)
        example = train_dataset.__getitem__(idx)
        print("Example:\n X: {}\n Y: {}".format(example[0], example[1]))


if __name__ == "__main__":
    main()
