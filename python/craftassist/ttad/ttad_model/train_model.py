import argparse
import logging
import logging.handlers
import os
import pickle

from pprint import pprint

from time import time

import torch.optim as optim

from acl_models import *

from ttad_model_wrapper import *


def main():
    parser = argparse.ArgumentParser(description="Train text to action dictionary model")
    parser.add_argument("-cuda", "--cuda", action="store_true", help="use GPU to train")
    parser.add_argument(
        "-mn",
        "--model_name",
        default="data/models/tree_model",
        type=str,
        metavar="S",
        help="torch formatted model file to save",
    )
    parser.add_argument(
        "-rp",
        "--rephrase_proba",
        default=0.4,
        type=float,
        metavar="D",
        help="proportion of rephrased training data 0 <= r <= 1",
    )
    # data sampling arguments
    parser.add_argument(
        "-rsm",
        "--resample_mode",
        default="action-type",
        type=str,
        metavar="S",
        help="re-sample train and valid data to match [none|action-type|tree-internal|tree-full] distribution",
    )
    parser.add_argument(
        "-rst",
        "--resample_type",
        default="humanbot",
        type=str,
        metavar="S",
        help="re-sample train and valid data to match [rephrased|prompts|humanbot|templated] distribution",
    )
    parser.add_argument(
        "-hbp",
        "--hard_buffer_proba",
        default=0.3,
        type=float,
        metavar="D",
        help="proportion of training batch examples from hard examples buffer 0 <= r <= 1",
    )
    parser.add_argument(
        "-hbs",
        "--hard_buffer_size",
        default=2048,
        type=int,
        metavar="N",
        help="proportion of training batch examples from hard examples buffer 0 <= r <= 1",
    )
    # file locations
    parser.add_argument(
        "-toe",
        "--train_on_everything",
        action="store_true",
        help="train on validation and test set too",
    )
    parser.add_argument(
        "-df",
        "--data_file",
        default="acl_data/all_data_newdct.json",
        type=str,
        metavar="S",
        help="location of json file with all taining, valid and test data",
    )
    parser.add_argument(
        "-atf",
        "--action_tree_file",
        default="acl_data/action_tree_newdct.json",
        type=str,
        metavar="S",
        help="action tree that covers all of the data (including OtherAction)",
    )
    parser.add_argument(
        "-ftmf",
        "--ft_model_file",
        default="acl_data/minecraft_newdct_ft_embeds.pth",
        type=str,
        metavar="S",
        help="pre-trained pre-computed subword-aware fasttext embeddings and vocabulary",
    )
    # model arguments
    parser.add_argument("-cn", "--collapse_nodes", action="store_true", help="parameter sharing")
    parser.add_argument(
        "-led",
        "--learn_embed_dim",
        default=0,
        type=int,
        metavar="N",
        help="concatenate learned embeddings to FastText pretained of dimension",
    )
    parser.add_argument(
        "-learn_embeds",
        "--learn_embeds",
        action="store_true",
        help="learn word embeddings from scratch",
    )
    parser.add_argument(
        "-rec",
        "--recursion",
        default="sentence-rec",
        type=str,
        help="type of recursion for the prediction model in"
        "[none|top-down|dfs-intern|dfs-all|sentence-rec]",
    )
    parser.add_argument(
        "-md", "--model_dim", default=256, type=int, metavar="N", help="model dimension"
    )
    parser.add_argument(
        "-sh",
        "--sent_heads",
        default=4,
        type=int,
        metavar="N",
        help="number of sentence attention heads",
    )
    parser.add_argument(
        "-th",
        "--tree_heads",
        default=4,
        type=int,
        metavar="N",
        help="number of tree context attention heads",
    )
    parser.add_argument(
        "-sec",
        "--sentence_encoder",
        default="bigru",
        type=str,
        help="type of sentence encoder [convolution|bigru]",
    )
    parser.add_argument(
        "-scl",
        "--sent_conv_layers",
        default=3,
        type=int,
        metavar="N",
        help="convolutional layers for sentence encoder",
    )
    parser.add_argument(
        "-scw",
        "--sent_conv_window",
        default=1,
        type=int,
        metavar="N",
        help="window size (2k+1) for sentence encoder convolutions",
    )
    parser.add_argument(
        "-sgl",
        "--sent_gru_layers",
        default=2,
        type=int,
        metavar="N",
        help="BiGRU layers for sentence encoder",
    )
    # optimization arguments
    parser.add_argument(
        "-bs", "--batch_size", default=128, type=int, metavar="N", help="batch size"
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=5e-3,
        type=float,
        metavar="D",
        help="learning rate factor",
    )
    parser.add_argument(
        "-le", "--learning_epochs", default=10, type=int, metavar="N", help="number of epochs"
    )
    parser.add_argument(
        "-nbpe",
        "--batches_per_epoch",
        default=1000,
        type=int,
        metavar="N",
        help="number of batches per epoch",
    )
    parser.add_argument(
        "-ls",
        "--label_smoothing",
        default=0.0,
        type=float,
        metavar="D",
        help="label smoothing regularizer",
    )
    parser.add_argument(
        "-do", "--dropout", default=0.3, type=float, metavar="D", help="dropout value in [0,1]"
    )
    parser.add_argument(
        "-sn",
        "--sentence_noise",
        default=0.0,
        type=float,
        metavar="D",
        help="train time probability of masking a word in [0,1]",
    )
    # continue training
    parser.add_argument("-lm", "--load_model", action="store_true")
    parser.add_argument(
        "-lmf",
        "--load_model_file",
        default="",
        type=str,
        metavar="S",
        help="torch formatted model file to load",
    )
    args = parser.parse_args()
    # TODO: add print_freq to arguments
    args.print_freq = 100
    # short-hand for no pre-trained word embeddings
    if args.learn_embed_dim < 0:
        args.learn_embed_dim *= -1
        args.learn_embeds = True
    pickle.dump(args, open("%s.args.pk" % (args.model_name,), "wb"))
    ### set up logging
    l_handler = logging.handlers.WatchedFileHandler(
        os.environ.get("LOGFILE", "logs/%s.log" % args.model_name.split("/")[-1])
    )
    l_format = logging.Formatter(fmt="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
    l_handler.setFormatter(l_format)
    l_root = logging.getLogger()
    l_root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    l_root.addHandler(l_handler)
    ### Training data loading and training code
    st_time = time()
    # load make data
    print("starting %.1f" % (time() - st_time,))
    a_tree_pred, a_tree_loss, w2i = make_model(
        args, args.ft_model_file, args.action_tree_file, args.cuda
    )
    optimizer = optim.Adagrad(a_tree_pred.parameters(), lr=args.learning_rate)
    print("made model %.1f" % (time() - st_time,))
    # load data
    data_loader = DataLoader(args)
    print("loaded training data:  %.1f" % (time() - st_time,))

    if args.load_model_file:
        print("loading model to test....")
        a_tree_pred.load_state_dict(torch.load(args.load_model_file))
        a_tree_pred.eval()

        # get numbers for validation set
        for valid_type in data_loader.data["valid"]:
            # accuracy
            acc1 = compute_accuracy(a_tree_pred, data_loader.data["valid"][valid_type], w2i, args)
            # accuracy per action type
            acc2 = compute_accuracy_per_action_type(
                a_tree_pred, data_loader.data["valid"][valid_type], w2i, args
            )
            # precision, recall, f1
            precision_recall_stats = compute_precision_recall(
                a_tree_pred, data_loader.data["valid"][valid_type][:10], w2i, args
            )
            print("----- VALID_FINAL_ACCURACIES %s %s" % (valid_type, str(acc1)))
            pprint(str(acc2))
            pprint(str(precision_recall_stats))
            print("-------" * 10)

        print("----- TEST")
        for test_type in data_loader.data["test"]:
            # for internal only accuracy -> give : only_internal=True
            # get accuracy
            acc1 = compute_accuracy(a_tree_pred, data_loader.data["test"][test_type], w2i, args)
            # get accuracy per action type
            acc2 = compute_accuracy_per_action_type(
                a_tree_pred, data_loader.data["test"][test_type], w2i, args
            )
            # precision, recall, f1 stats
            precision_recall_stats = compute_precision_recall(
                a_tree_pred, data_loader.data["test"][test_type], w2i, args
            )

            print("----- TEST_FINAL_ACCURACIES %s %s" % (test_type, str(acc1)))
            pprint(str(acc2))
            pprint(str(precision_recall_stats))
            print("------" * 10)

    else:
        # start training
        for e in range(args.learning_epochs):
            model_file_name = "%s_%d.pth" % (args.model_name, e)
            # train on current slice
            a_tree_pred.train()
            print("----- TRAIN GENERATED", e)
            logging.info("----- TRAIN GENERATED %d" % (e,))
            run_epoch(
                data_loader,
                a_tree_pred,
                a_tree_loss,
                w2i,
                args,
                mode="train",
                data_type="",
                optimizer=optimizer,
            )
            # DEBUG
            torch.save(a_tree_pred.to("cpu").state_dict(), model_file_name)
            a_tree_pred.to("cuda:0")
            a_tree_pred.eval()

            # compute accuracy on hard buffer
            hard_buf_acc = compute_accuracy(a_tree_pred, data_loader.hard_buffer[:512], w2i, args)
            logging.info("---- HARD_BUFFER_ACCURACY %s %s " % (e, hard_buf_acc))

            # compute accuracy on valid
            data_loader.reset_valid()
            for valid_type in data_loader.data["valid"]:
                print("----- VALID %s" % valid_type, e)
                logging.info("----- VALID %s %d" % (valid_type, e))
                run_epoch(
                    data_loader,
                    a_tree_pred,
                    a_tree_loss,
                    w2i,
                    args,
                    mode="valid",
                    data_type=valid_type,
                    optimizer=None,
                )
                acc = compute_accuracy(
                    a_tree_pred, data_loader.data["valid"][valid_type][:512], w2i, args
                )
                logging.info("----- VALID_ACCURACIES %d %s %.2f" % (e, valid_type, acc))

        # compute acuracy on test
        print("----- TEST")
        for test_type in data_loader.data["test"]:
            print(test_type)
            acc = compute_accuracy(a_tree_pred, data_loader.data["test"][test_type], w2i, args)
            print(acc)
            logging.info("----- TEST_ACCURACIES %s %.2f" % (test_type, acc))


if __name__ == "__main__":
    main()
