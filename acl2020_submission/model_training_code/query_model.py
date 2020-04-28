import json
import math
import pickle

import torch

from transformers import AutoModel, AutoTokenizer, BertConfig

from utils_parsing import *
from utils_caip import *
from train_model import *


class TTADBertModel(object):
    def __init__(self, model_dir, data_dir, model_name="caip_test_model"):
        model_name = model_dir + model_name
        args = pickle.load(open(model_name + "_args.pk", "rb"))

        args.data_dir = data_dir

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_encoder_name)
        full_tree, tree_i2w = json.load(open(model_name + "_tree.json"))
        self.dataset = CAIPDataset(
            self.tokenizer, args, prefix="", full_tree_voc=(full_tree, tree_i2w)
        )

        enc_model = AutoModel.from_pretrained(args.pretrained_encoder_name)
        bert_config = BertConfig.from_pretrained("bert-base-uncased")
        bert_config.is_decoder = True
        bert_config.vocab_size = len(tree_i2w) + 8

        bert_config.num_hidden_layers = args.num_decoder_layers
        dec_with_loss = DecoderWithLoss(bert_config, args, self.tokenizer)
        self.encoder_decoder = EncoderDecoderWithLoss(enc_model, dec_with_loss, args)
        map_location = None if torch.cuda.is_available() else torch.device("cpu")
        self.encoder_decoder.load_state_dict(
            torch.load(model_name + ".pth", map_location=map_location), strict=False
        )
        self.encoder_decoder = (
            self.encoder_decoder.cuda()
            if torch.cuda.is_available()
            else self.encoder_decoder.cpu()
        )
        self.encoder_decoder.eval()

    def parse(self, chat, noop_thres=0.95, beam_size=5, well_formed_pen=1e2):
        btr = beam_search(
            chat, self.encoder_decoder, self.tokenizer, self.dataset, beam_size, well_formed_pen
        )
        if btr[0][0].get("dialogue_type", "NONE") == "NOOP" and math.exp(btr[0][1]) < noop_thres:
            tree = btr[1][0]
        else:
            tree = btr[0][0]
        return tree
