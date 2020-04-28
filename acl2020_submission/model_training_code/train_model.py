import argparse
import functools
import json
import logging
import logging.handlers
import os
import pickle

from os.path import isfile
from os.path import join as pjoin
from glob import glob
from tqdm import tqdm
from time import time

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModel, AutoTokenizer, BertConfig

from utils_parsing import *
from utils_caip import *


# training loop (all epochs at once)
def train(model, dataset, tokenizer, args):
    # make data sampler
    train_sampler = RandomSampler(dataset)
    model_collate_fn = functools.partial(caip_collate, tokenizer=tokenizer)
    train_dataloader = DataLoader(
        dataset, sampler=train_sampler, batch_size=args.batch_size, collate_fn=model_collate_fn
    )
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
    # make optimizer
    optimizer = OptimWarmupEncoderDecoder(model, args)
    # training loop
    for e in range(args.num_epochs):
        loc_steps = 0
        loc_loss = 0.0
        loc_int_acc = 0.0
        loc_span_acc = 0.0
        loc_full_acc = 0.0
        tot_steps = 0
        tot_loss = 0.0
        tot_accuracy = 0.0
        st_time = time()
        for step, batch in enumerate(epoch_iterator):
            batch_examples = batch[-1]
            batch_tensors = [
                t.to(model.decoder.lm_head.predictions.decoder.weight.device) for t in batch[:4]
            ]
            x, x_mask, y, y_mask = batch_tensors
            outputs = model(x, x_mask, y, y_mask)
            loss = outputs["loss"]
            # backprop
            loss.backward()
            if step % args.param_update_freq == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                model.zero_grad()
            # compute accuracy and add hard examples
            lm_acc, sp_acc, full_acc = compute_accuracy(outputs, y)
            if "hard" in dataset.dtypes:
                if e > 0 or tot_steps > 2 * args.decoder_warmup_steps:
                    for acc, exple in zip(lm_acc, batch_examples):
                        if not acc.item():
                            if step % 200 == 100:
                                print("ADDING HE:", step, exple[0])
                            dataset.add_hard_example(exple)
            # book-keeping
            loc_int_acc += lm_acc.sum().item() / lm_acc.shape[0]
            loc_span_acc += sp_acc.sum().item() / sp_acc.shape[0]
            loc_full_acc += full_acc.sum().item() / full_acc.shape[0]
            tot_accuracy += full_acc.sum().item() / full_acc.shape[0]
            loc_loss += loss.item()
            loc_steps += 1
            tot_loss += loss.item()
            tot_steps += 1
            if step % 400 == 0:
                print(
                    "{:2d} - {:5d} \t L: {:.3f} A: {:.3f} \t {:.2f}".format(
                        e, step, loc_loss / loc_steps, loc_full_acc / loc_steps, time() - st_time
                    )
                )
                logging.info(
                    "{:2d} - {:5d} \t L: {:.3f} A: {:.3f} \t {:.2f}".format(
                        e, step, loc_loss / loc_steps, loc_full_acc / loc_steps, time() - st_time
                    )
                )
                loc_loss = 0
                loc_steps = 0
                loc_int_acc = 0.0
                loc_span_acc = 0.0
                loc_full_acc = 0.0
    return (tot_loss / tot_steps, tot_accuracy / tot_steps)


# same as training loop without back-propagation
def validate(model, dataset, tokenizer, args):
    # make data sampler
    train_sampler = SequentialSampler(dataset)
    model_collate_fn = functools.partial(caip_collate, tokenizer=tokenizer)
    train_dataloader = DataLoader(
        dataset, sampler=train_sampler, batch_size=args.batch_size, collate_fn=model_collate_fn
    )
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
    # training loop
    tot_steps = 0
    tot_loss = 0.0
    tot_int_acc = 0.0
    tot_span_acc = 0.0
    tot_accu = 0.0
    for step, batch in enumerate(epoch_iterator):
        batch_tensors = [
            t.to(model.decoder.lm_head.predictions.decoder.weight.device) for t in batch[:4]
        ]
        x, x_mask, y, y_mask = batch_tensors
        outputs = model(x, x_mask, y, y_mask)
        loss = outputs["loss"]
        # compute accuracy and add hard examples
        lm_acc, sp_acc, full_acc = compute_accuracy(outputs, y)
        # book-keeping
        tot_int_acc += lm_acc.sum().item() / lm_acc.shape[0]
        tot_span_acc += sp_acc.sum().item() / sp_acc.shape[0]
        tot_accu += full_acc.sum().item() / full_acc.shape[0]
        tot_loss += loss.item()
        tot_steps += 1
    return (
        tot_loss / tot_steps,
        tot_int_acc / tot_steps,
        tot_span_acc / tot_steps,
        tot_accu / tot_steps,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default="reformatted_ttad_data", type=str, help="train/valid/test data"
    )
    parser.add_argument(
        "--output_dir", default="caip_model_dir", type=str, help="Where we save the model"
    )
    parser.add_argument("--model_name", default="caip_parser", type=str, help="Model name")
    parser.add_argument(
        "--tree_voc_file",
        default="caip_tree_voc_final.json",
        type=str,
        help="Pre-computed grammar and output vocabulary",
    )
    # model arguments
    parser.add_argument(
        "--pretrained_encoder_name",
        default="distilbert-base-uncased",
        type=str,
        help="Pretrained text encoder "
        "See full list at https://huggingface.co/transformers/pretrained_models.html",
    )
    parser.add_argument(
        "--num_decoder_layers",
        default=6,
        type=int,
        help="Number of transformer layers in the decoder",
    )
    parser.add_argument(
        "--num_highway", default=1, type=int, help="Number of highway layers in the mapping model"
    )
    # optimization arguments
    parser.add_argument(
        "--optimizer", default="adam", type=str, help="Optimizer in [adam|adagrad]"
    )
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
    parser.add_argument("--param_update_freq", default=1, type=int, help="Group N batch updates")
    parser.add_argument("--num_epochs", default=2, type=int, help="Number of training epochs")
    parser.add_argument(
        "--examples_per_epoch", default=-1, type=int, help="Number of training examples per epoch"
    )
    parser.add_argument(
        "--train_encoder", action="store_true", help="Whether to finetune the encoder"
    )
    parser.add_argument(
        "--encoder_warmup_steps",
        default=1,
        type=int,
        help="Learning rate warmup steps for the encoder",
    )
    parser.add_argument(
        "--encoder_learning_rate", default=0.0, type=float, help="Learning rate for the encoder"
    )
    parser.add_argument(
        "--decoder_warmup_steps",
        default=1000,
        type=int,
        help="Learning rate warmup steps for the decoder",
    )
    parser.add_argument(
        "--decoder_learning_rate", default=1e-4, type=float, help="Learning rate for the decoder"
    )
    parser.add_argument(
        "--lambda_span_loss",
        default=0.5,
        type=float,
        help="Weighting between node and span prediction losses",
    )
    parser.add_argument(
        "--node_label_smoothing",
        default=0.0,
        type=float,
        help="Label smoothing for node prediction",
    )
    parser.add_argument(
        "--span_label_smoothing",
        default=0.0,
        type=float,
        help="Label smoothing for span prediction",
    )
    parser.add_argument(
        "--dtype_samples",
        default='[["templated", 1.0]]',
        type=str,
        help="Sampling probabilities for handling different data types",
    )
    parser.add_argument(
        "--rephrase_proba", default=-1.0, type=float, help="Only specify probablility of rephrases"
    )
    parser.add_argument(
        "--word_dropout",
        default=0.0,
        type=float,
        help="Probability of replacing input token with [UNK]",
    )
    parser.add_argument(
        "--encoder_dropout", default=0.0, type=float, help="Apply dropout to encoder output"
    )
    args = parser.parse_args()
    # HACK: allows us to give rephrase proba only instead of full dictionary
    if args.rephrase_proba > 0:
        args.dtype_samples = json.dumps(
            [["templated", 1.0 - args.rephrase_proba], ["rephrases", args.rephrase_proba]]
        )
    # set up logging
    l_handler = logging.handlers.WatchedFileHandler(
        os.environ.get("LOGFILE", "logs/%s.log" % args.model_name.split("/")[-1])
    )
    l_format = logging.Formatter(fmt="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
    l_handler.setFormatter(l_format)
    l_root = logging.getLogger()
    l_root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    l_root.addHandler(l_handler)
    # make dataset
    if isfile(args.tree_voc_file):
        print("loading grammar")
        logging.info("loading grammar")
        full_tree, tree_i2w = json.load(open(args.tree_voc_file))
    else:
        print("making grammar")
        logging.info("making grammar")
        data = {"train": {}, "valid": {}, "test": {}}
        for spl in data:
            for fname in glob(pjoin(args.data_dir, "{}/*.json".format(spl))):
                print(fname)
                data[spl][fname.split("/")[-1][:-5]] = json.load(open(fname))
        full_tree, tree_i2w = make_full_tree(
            [
                (d_list, 1.0)
                for spl, dtype_dict in data.items()
                for dtype, d_list in dtype_dict.items()
            ]
        )
        json.dump((full_tree, tree_i2w), open(args.tree_voc_file, "w"))
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_encoder_name)
    logging.info("loading data")
    train_dataset = CAIPDataset(
        tokenizer,
        args,
        prefix="train",
        sampling=True,
        word_noise=args.word_dropout,
        full_tree_voc=(full_tree, tree_i2w),
    )
    # make model
    logging.info("making model")
    enc_model = AutoModel.from_pretrained(args.pretrained_encoder_name)
    bert_config = BertConfig.from_pretrained("bert-base-uncased")
    bert_config.is_decoder = True
    bert_config.vocab_size = len(tree_i2w) + 8  # special tokens
    bert_config.num_hidden_layers = args.num_decoder_layers
    dec_with_loss = DecoderWithLoss(bert_config, args, tokenizer)
    encoder_decoder = EncoderDecoderWithLoss(enc_model, dec_with_loss, args)
    # train_model
    logging.info("training model")
    encoder_decoder = encoder_decoder.cuda()
    encoder_decoder.train()
    loss, accu = train(encoder_decoder, train_dataset, tokenizer, args)
    # save model
    json.dump(
        (full_tree, tree_i2w), open(pjoin(args.output_dir, args.model_name + "_tree.json"), "w")
    )
    pickle.dump(args, open(pjoin(args.output_dir, args.model_name + "_args.pk"), "wb"))
    torch.save(encoder_decoder.state_dict(), pjoin(args.output_dir, args.model_name + ".pth"))
    # evaluate model
    _ = encoder_decoder.eval()
    logging.info("evaluating model")
    valid_template = CAIPDataset(
        tokenizer, args, prefix="valid", dtype="templated", full_tree_voc=(full_tree, tree_i2w)
    )
    l, _, _, a = validate(encoder_decoder, valid_template, tokenizer, args)
    print("evaluating on templated valid: \t Loss: {:.4f} \t Accuracy: {:.4f}".format(l, a))
    logging.info("evaluating on templated valid: \t Loss: {:.4f} \t Accuracy: {:.4f}".format(l, a))
    valid_rephrase = CAIPDataset(
        tokenizer, args, prefix="valid", dtype="rephrases", full_tree_voc=(full_tree, tree_i2w)
    )
    l, _, _, a = validate(encoder_decoder, valid_rephrase, tokenizer, args)
    print("evaluating on rephrases valid: \t Loss: {:.4f} \t Accuracy: {:.4f}".format(l, a))
    logging.info("evaluating on rephrases valid: \t Loss: {:.4f} \t Accuracy: {:.4f}".format(l, a))
    valid_prompts = CAIPDataset(
        tokenizer, args, prefix="valid", dtype="prompts", full_tree_voc=(full_tree, tree_i2w)
    )
    l, _, _, a = validate(encoder_decoder, valid_prompts, tokenizer, args)
    print("evaluating on prompts valid: \t Loss: {:.4f} \t Accuracy: {:.4f}".format(l, a))
    logging.info("evaluating on prompts valid: \t Loss: {:.4f} \t Accuracy: {:.4f}".format(l, a))
    valid_humanbot = CAIPDataset(
        tokenizer, args, prefix="valid", dtype="humanbot", full_tree_voc=(full_tree, tree_i2w)
    )
    l, _, _, a = validate(encoder_decoder, valid_humanbot, tokenizer, args)
    print("evaluating on humanbot valid: \t Loss: {:.4f} \t Accuracy: {:.4f}".format(l, a))
    logging.info("evaluating on humanbot valid: \t Loss: {:.4f} \t Accuracy: {:.4f}".format(l, a))


if __name__ == "__main__":
    main()
