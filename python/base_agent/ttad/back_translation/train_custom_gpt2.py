from transformers import AutoConfig, AutoTokenizer
from dataset import *
from modeling_gpt2 import *
from transformer import *
from torch.utils.data import DataLoader
import argparse
from os.path import join as pjoin
import torch
from train_utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="/checkpoint/rebeccaqian/datasets/fairseq/08_04/",
        type=str,
        help="train/valid/test data",
    )
    parser.add_argument(
        "--output_dir",
        default="/checkpoint/rebeccaqian/backtranslation/{}/models/".format(str(date.today())),
        type=str,
        help="directory to save checkpoint models",
    )
    parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--num_epochs", default=20, type=int, help="number of epochs")
    args = parser.parse_args()
    config_decoder = AutoConfig.from_pretrained("gpt2", is_decoder=True)
    config_encoder = AutoConfig.from_pretrained("gpt2", is_decoder=False)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    enc_model = GPT2Model(config=config_encoder)
    dec_model = GPT2LMHeadModel.from_pretrained("gpt2", config=config_decoder)
    encoder_decoder = EncoderDecoder(enc_model, dec_model)
    # freeze decoder weights
    decoder_params = dec_model.named_parameters()
    for name, param in decoder_params:
        param.requires_grad = False

    train_prog_data, train_chat_data = load_paired_dataset(args.data_dir, "train")
    train_dataset = Tree2TextDataset(train_prog_data, train_chat_data)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
    valid_prog_data, valid_chat_data = load_paired_dataset(args.data_dir, "valid")
    valid_dataset = Tree2TextDataset(valid_prog_data, valid_chat_data)
    optimizer = torch.optim.Adam(encoder_decoder.parameters(), lr=args.lr)

    for epoch in range(args.num_epochs):
        print("Epoch: {}".format(epoch))
        enc_model.train()
        dec_model.train()

        for i, batch in enumerate(train_loader):
            trees, text = batch
            optimizer.zero_grad()
            text_idx_ls = [tokenizer.encode(cmd) for cmd in text]
            tree_idx_ls = [tokenizer.encode(tree) for tree in trees]
            x, x_mask, y, y_mask = collate(tree_idx_ls, text_idx_ls, tokenizer)
            outputs = encoder_decoder(x, x_mask, y, y_mask)
            loss, predictions = outputs[:2]
            # Taking the first row in the batch as logging example
            predicted_example = predictions.max(dim=-1)[1][:, :-1]
            predicted_sentence = tokenizer.decode(predicted_example[0])
            print(
                "sample predictions:\n X: {} Y: {} predicted: {}".format(
                    trees[0], text[0], predicted_sentence
                )
            )
            loss.backward()
            optimizer.step()
            print("Iteration: {} Loss: {}".format(i, loss))

        torch.save(
            encoder_decoder.state_dict(),
            pjoin("{}encoder|(ep=={}).pth".format(args.output_dir, epoch)),
        )
        # Evaluating model
        encoder_decoder.eval()
        print("Evaluating model")
        validate(encoder_decoder, valid_dataset, tokenizer)


if __name__ == "__main__":
    main()
