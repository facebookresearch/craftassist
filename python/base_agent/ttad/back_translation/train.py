from transformers import AutoConfig, GPT2Tokenizer, BertTokenizer
from os.path import isdir
from dataset import *
from transformers.modeling_gpt2 import *
from transformer import *
from torch.utils.data import DataLoader
from datetime import date
import argparse
import os
import sys
import logging
import logging.handlers
from time import time


BASE_AGENT_ROOT = os.path.join(os.path.dirname(__file__), "../..")
sys.path.append(BASE_AGENT_ROOT)


# make sure GPT2 appends EOS in begin and end
def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    return outputs


def compute_accuracy(outputs, y, tokenizer):
    """Compute model accuracy given predictions and targets. Used in validation.
    """
    # Do not include BOS token
    target_tokens = y[:, 1:]
    predicted_tokens = outputs.max(dim=-1)[1][:, :-1]
    acc = (predicted_tokens == target_tokens).sum(dim=1) == target_tokens.shape[-1]
    return acc


def validate(model, dataset, text_tokenizer, tree_tokenizer):
    valid_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=10, shuffle=False)
    tot_acc = 0.0
    tot_loss = 0.0
    steps = 0
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            trees, text = batch
            text_idx_ls = [
                text_tokenizer.encode(cmd.strip(), add_special_tokens=True) for cmd in text
            ]
            tree_idx_ls = [tree_tokenizer.encode(tree, add_special_tokens=True) for tree in trees]
            x, x_mask, y, y_mask = collate(
                tree_idx_ls, text_idx_ls, tree_tokenizer, text_tokenizer
            )
            y_copy = y.copy()
            y_mask_copy = y_mask.copy()
            labels = [
                [-100 if mask == 0 else token for mask, token in mask_and_tokens]
                for mask_and_tokens in [
                    zip(mask, label) for mask, label in zip(y_mask_copy, y_copy)
                ]
            ]
            labels = torch.tensor(labels)
            x = torch.tensor(x)
            x_mask = torch.tensor(x_mask)
            y = torch.tensor(y)
            y_mask = torch.tensor(y_mask)
            out = model(x, x_mask, y, y_mask, labels)
            loss, predictions = out[:2]
            # Taking the first row in the batch as logging example
            predicted_example = predictions.max(dim=-1)[1][:, :-1]
            predicted_sentence = text_tokenizer.decode(predicted_example[0])
            logging.info(
                "sample predictions:\n X: {} \n Y: {} \n predicted: {}".format(
                    trees[0], text[0], predicted_sentence
                )
            )
            print(
                "sample predictions:\n X: {} \n Y: {} \n predicted: {}".format(
                    trees[0], text[0], predicted_sentence
                )
            )
            lm_acc = compute_accuracy(predictions, y, text_tokenizer)
            acc = lm_acc.sum().item() / lm_acc.shape[0]
            tot_acc += acc
            tot_loss += loss.item()
            steps += 1
    logging.info("Valid accuracy: {} Loss: {}".format(tot_acc / steps, tot_loss / steps))
    print("Valid accuracy: {} Loss: {}".format(tot_acc / steps, tot_loss / steps))


def generate_model_name(args):
    # unix time in seconds, used as a unique identifier
    time_now = round(time())
    name = ""
    args_keys = {
        "batch_size": "batch",
        "lr": "lr",
        "train_decoder": "finetune",
        "encoder_size": "size",
    }
    for k, v in vars(args).items():
        if k in args_keys:
            name += "{param}={value}|".format(param=args_keys[k], value=v)
    # In case we want additional identification for the model, eg. test run
    name += "{time}|".format(time=time_now)
    name += args.model_identifier
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="/checkpoint/rebeccaqian/datasets/backtranslation/09_11/mixed_spans/",
        type=str,
        help="train/valid/test data",
    )
    parser.add_argument(
        "--output_dir",
        default="/checkpoint/rebeccaqian/backtranslation/{}/".format(str(date.today())),
        type=str,
        help="directory to save checkpoint models",
    )
    parser.add_argument(
        "--train_decoder",
        default=False,
        action="store_true",
        help="whether to finetune the decoder",
    )
    parser.add_argument(
        "--model_identifier",
        default="ttad_bert2gpt2",
        type=str,
        help="optional identifier string used in filenames",
    )
    parser.add_argument("--encoder_size", default="medium", type=str, help="size of encoder")
    parser.add_argument(
        "--num_hidden_layers",
        default=12,
        type=int,
        help="number of hidden layers in BERT Transformer",
    )
    parser.add_argument(
        "--hidden_dropout_prob",
        default=0.1,
        type=int,
        help="dropout probabilitiy for all fully connected encoder layers",
    )
    parser.add_argument(
        "--data_type", default="annotated", type=str, help="name of dataset to load from"
    )
    parser.add_argument(
        "--use_lm", default=False, action="store_true", help="whether to use decoder as lm"
    )
    parser.add_argument("--lr", default=0.00001, type=float, help="learning rate")
    parser.add_argument("--batch_size", default=48, type=int, help="batch size")
    parser.add_argument("--num_epochs", default=30, type=int, help="number of epochs")
    args = parser.parse_args()
    model_identifier = generate_model_name(args)
    l_handler = logging.handlers.WatchedFileHandler(
        "{}/{}.log".format(args.output_dir, model_identifier)
    )
    l_format = logging.Formatter(fmt="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
    l_handler.setFormatter(l_format)
    l_root = logging.getLogger()
    l_root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    l_root.addHandler(l_handler)
    logging.info(args)
    print(args)

    if args.encoder_size == "large":
        encoder_name = "bert-large-uncased"
        decoder_name = "gpt2-medium"
    else:
        encoder_name = "bert-base-uncased"
        decoder_name = "gpt2"

    config_decoder = AutoConfig.from_pretrained(decoder_name, is_decoder=True)
    config_decoder.add_cross_attention = True
    encoder_name
    config_encoder = AutoConfig.from_pretrained(encoder_name, is_decoder=False)
    bert_tokenizer = BertTokenizer.from_pretrained(encoder_name)
    # CLS token will work as BOS token
    bert_tokenizer.bos_token = bert_tokenizer.cls_token
    # SEP token will work as EOS token
    bert_tokenizer.eos_token = bert_tokenizer.sep_token

    tokenizer = GPT2Tokenizer.from_pretrained(decoder_name)
    tokenizer.pad_token = tokenizer.unk_token
    GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
    enc_model = BertModel(config=config_encoder)
    dec_model = GPT2LMHeadModel.from_pretrained(decoder_name, config=config_decoder)
    dec_model.config.decoder_start_token_id = tokenizer.bos_token_id
    dec_model.config.eos_token_id = tokenizer.eos_token_id
    dec_model.resize_token_embeddings(len(tokenizer))
    decoder_params = dec_model.named_parameters()
    if not args.train_decoder:
        # enumerate all decoder params, 244
        decoder_params = dec_model.named_parameters()
        for name, param in decoder_params:
            # train decoder attention weights, 94
            if "cross_attn" in name or "crossattention" in name:
                param.requires_grad = True
            else:
                # freeze other layer weights
                param.requires_grad = False

    train_prog_data, train_chat_data = load_paired_dataset(args.data_dir, "train")
    train_dataset = Tree2TextDataset(train_prog_data, train_chat_data)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_prog_data, valid_chat_data = load_paired_dataset(args.data_dir, "valid")
    valid_dataset = Tree2TextDataset(valid_prog_data, valid_chat_data)
    encoder_decoder = EncoderDecoder(enc_model, dec_model)
    optimizer = torch.optim.Adam(encoder_decoder.parameters(), lr=args.lr)

    for epoch in range(args.num_epochs):
        logging.info("Epoch: {}".format(epoch))
        print("Epoch: {}".format(epoch))

        encoder_decoder.train()

        for i, batch in enumerate(train_loader):
            trees, text = batch
            optimizer.zero_grad()
            text_idx_ls = [tokenizer.encode(cmd.strip(), add_special_tokens=True) for cmd in text]
            tree_idx_ls = [bert_tokenizer.encode(tree, add_special_tokens=True) for tree in trees]
            x, x_mask, y, y_mask = collate(tree_idx_ls, text_idx_ls, bert_tokenizer, tokenizer)
            y_copy = y.copy()
            y_mask_copy = y_mask.copy()
            labels = [
                [-100 if mask == 0 else token for mask, token in mask_and_tokens]
                for mask_and_tokens in [
                    zip(mask, label) for mask, label in zip(y_mask_copy, y_copy)
                ]
            ]
            labels = torch.tensor(labels)
            y = torch.tensor(y)
            y_mask = torch.tensor(y_mask)
            x = torch.tensor(x)
            x_mask = torch.tensor(x_mask)
            outputs = encoder_decoder(x, x_mask, y, y_mask, labels, args.use_lm)
            loss, predictions = outputs[:2]
            # Taking the first row in the batch as logging example
            predicted_example = predictions.max(dim=-1)[1][:, :-1]
            predicted_sentence = tokenizer.decode(predicted_example[0])
            loss.backward()
            optimizer.step()
            if i % 20 == 0:
                logging.info(
                    "sample predictions:\n X: {} \n Y: {} \n predicted: {}".format(
                        trees[0], text[0], predicted_sentence
                    )
                )
                print(
                    "sample predictions:\n X: {} \n Y: {} \n predicted: {}".format(
                        trees[0], text[0], predicted_sentence
                    )
                )
                logging.info("Iteration: {} Loss: {}".format(i, loss))
                print("Iteration: {} Loss: {}".format(i, loss))
        enc_dirpath = "{}encoder/".format(args.output_dir)
        if not isdir(enc_dirpath):
            os.mkdir(enc_dirpath)
        enc_checkpoint_path = enc_dirpath + "{}(ep=={})/".format(model_identifier, epoch)
        if not isdir(enc_checkpoint_path):
            os.mkdir(enc_checkpoint_path)
        dec_dirpath = "{}decoder/".format(args.output_dir)
        if not isdir(dec_dirpath):
            os.mkdir(dec_dirpath)
        dec_checkpoint_path = dec_dirpath + "{}(ep=={})/".format(model_identifier, epoch)
        if not isdir(dec_checkpoint_path):
            os.mkdir(dec_checkpoint_path)
        enc_model.save_pretrained(enc_checkpoint_path)
        dec_model.save_pretrained(dec_checkpoint_path)
        # Evaluating model
        encoder_decoder.eval()
        logging.info("Evaluating model")
        validate(encoder_decoder, valid_dataset, tokenizer, bert_tokenizer)


if __name__ == "__main__":
    main()
