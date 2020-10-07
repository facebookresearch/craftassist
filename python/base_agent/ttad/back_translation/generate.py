from transformers import GPT2Tokenizer, BertTokenizer
from dataset import *

# from transformers.modeling_gpt2 import *
from huggingface_modeling_gpt2 import *
from transformer import *
import argparse
import numpy as np
import torch.nn.functional as F

# make sure GPT2 appends EOS in begin and end
def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    return outputs


class QueryModel:
    def __init__(self, model, encoder_tokenizer, decoder_tokenizer):
        self.model = model
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer

    def top_k_sampling(self, y, y_mask, x_reps, x_mask, ntok):
        orig_mask = y_mask
        for _ in range(ntok):
            out = self.model.decoder(
                input_ids=y,
                attention_mask=y_mask,
                encoder_hidden_states=x_reps,
                encoder_attention_mask=x_mask,
                # use_lm=True,
            )[0]
            logits = out[:, -1, :]
            indices_to_remove = logits < torch.topk(logits, 10)[0][..., -1, None]
            logits[indices_to_remove] = np.NINF
            next_tok = torch.multinomial(nn.Softmax(dim=-1)(logits), num_samples=1).squeeze(1)
            y = torch.cat([y, next_tok.unsqueeze(-1)], dim=-1)
            y_mask = F.pad(orig_mask, pad=(0, y.shape[-1] - 1), mode="constant", value=1)
            if next_tok == self.decoder_tokenizer.eos_token_id:
                return y
        return y

    def generate(self, tree_input, top_k=0, beam_size=0):
        model_device = self.model.encoder.device
        text_idx_ls = [[self.decoder_tokenizer.eos_token_id]]
        tree_idx_ls = [self.encoder_tokenizer.encode(tree_input, add_special_tokens=True)]
        batch = collate(tree_idx_ls, text_idx_ls, self.encoder_tokenizer, self.decoder_tokenizer)
        batch = [torch.tensor(t) for t in batch[:4]]
        batch = [t.to(model_device) for t in batch[:4]]
        x, x_mask, y, y_mask = batch
        x_reps = self.model.encoder(input_ids=x, attention_mask=x_mask)[0].detach()

        if top_k > 0:
            out = self.top_k_sampling(y, y_mask, x_reps, x_mask, top_k)
            return self.decoder_tokenizer.decode(out[0])

        if beam_size > 1:
            # NOTE: beam search is WIP
            x_mask = x_mask.expand(beam_size, -1)
            x_reps = x_reps.expand(beam_size, -1, -1)
            y = torch.LongTensor(
                [[self.decoder_tokenizer.eos_token_id] for _ in range(beam_size)]
            ).to(model_device)
            beam_scores = torch.Tensor([-1e9 for _ in range(beam_size)]).to(model_device)  # B
            beam_scores[0] = 0
            finished = [False for _ in range(beam_size)]
        else:
            # defaults to greedy
            y = torch.LongTensor([[self.decoder_tokenizer.eos_token_id]]).to(model_device)
            beam_scores = torch.Tensor([-1e9]).to(model_device)
            beam_scores[0] = 0
            preds = [
                [self.decoder_tokenizer._convert_id_to_token(self.decoder_tokenizer.eos_token_id)]
            ]
            finished = [False]
        pad_scores = torch.Tensor([-1e9] * self.decoder_tokenizer.vocab_size).to(model_device)
        for i in range(20):
            outputs = self.model.decoder(
                input_ids=y,
                attention_mask=y_mask,
                encoder_hidden_states=x_reps,
                encoder_attention_mask=x_mask,
                # use_lm=True,
            )[0]
            predicted_scores = outputs[:, -1, :]

            for i, fshed in enumerate(finished):
                if fshed:
                    predicted_scores[i] = pad_scores
            total_scores = predicted_scores + beam_scores[:, None]
            linearized_scores = total_scores.view(-1)
            sorted_scores, sorted_ids = linearized_scores.sort(dim=-1, descending=True)
            # all 0 for now
            seq_ids = sorted_ids // total_scores.shape[-1]
            s_word_ids = sorted_ids % total_scores.shape[-1]
            # just taking first and only element for now
            beam_scores = sorted_scores[:1]
            # beam size of 1
            beam_ids = seq_ids[:1]
            word_ids = s_word_ids[:1]
            words = [self.decoder_tokenizer.decode(word_ids)]
            # add next token
            y = torch.cat([y[beam_ids], word_ids[:, None]], dim=1)
            # find out whether sequence is finished; currently only one sequence
            pre_finished = [finished[b_id.item()] for b_id in beam_ids]
            new_finished = [
                w_id.item() == self.decoder_tokenizer.eos_token_id for w_id in word_ids
            ]
            finished = [p or n for p, n in zip(pre_finished, new_finished)]
            n_mask = 1 - torch.Tensor(finished).type_as(y_mask)
            y_mask = torch.cat([y_mask[beam_ids], n_mask[:, None]], dim=1)

            preds = [preds[0] + [(words[0])]]
            # check whether sequence has reached EOS; currently only one sequence
            if finished[0]:
                break
        return " ".join(preds[0])


parser = argparse.ArgumentParser()
parser.add_argument(
    "--pretrained_encoder_path",
    default="python/craftassist/models/backtranslation/encoder/",
    type=str,
    help="path to binarized model and config of saved encoder",
)
parser.add_argument(
    "--pretrained_decoder_path",
    default="python/craftassist/models/backtranslation/decoder/",
    type=str,
    help="path to binarized model and config of saved decoder",
)
args = parser.parse_args()
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_tokenizer.bos_token = bert_tokenizer.cls_token
bert_tokenizer.eos_token = bert_tokenizer.sep_token

gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_tokenizer.pad_token = gpt2_tokenizer.unk_token
GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens

enc_model = BertModel.from_pretrained(args.pretrained_encoder_path)
dec_model = GPT2LMHeadModel.from_pretrained(args.pretrained_decoder_path)
encoder_decoder = EncoderDecoder(enc_model, dec_model)
generator = QueryModel(encoder_decoder, bert_tokenizer, gpt2_tokenizer)
