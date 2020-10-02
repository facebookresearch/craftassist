from transformers import GPT2Tokenizer, BertTokenizer
from dataset import *
from transformers.modeling_gpt2 import *
from transformer import *
import argparse

# make sure GPT2 appends EOS in begin and end
def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    return outputs


def generate(tree_input, encoder_decoder, bert_tokenizer, gpt2_tokenizer):
    model_device = encoder_decoder.encoder.device
    text_idx_ls = [[gpt2_tokenizer.eos_token_id]]
    tree_idx_ls = [bert_tokenizer.encode(tree_input, add_special_tokens=True)]
    batch = collate(tree_idx_ls, text_idx_ls, bert_tokenizer, gpt2_tokenizer)
    batch = [torch.tensor(t) for t in batch[:4]]
    batch = [t.to(model_device) for t in batch[:4]]
    x, x_mask, y, y_mask = batch
    x_reps = encoder_decoder.encoder(input_ids=x, attention_mask=x_mask)[0].detach()
    # start decoding
    y = torch.LongTensor([[gpt2_tokenizer.eos_token_id]]).to(model_device)
    probs = torch.Tensor([-1e9]).to(model_device)
    probs[0] = 0
    preds = [[gpt2_tokenizer._convert_id_to_token(gpt2_tokenizer.eos_token_id)]]
    finished = [False]
    pad_scores = torch.Tensor([-1e9] * gpt2_tokenizer.vocab_size).to(model_device)
    for i in range(100):
        outputs = encoder_decoder.decoder(
            input_ids=y,
            attention_mask=y_mask,
            encoder_hidden_states=x_reps,
            encoder_attention_mask=x_mask,
        )[0]
        predicted_scores = outputs[:, -1, :]
        for i, fshed in enumerate(finished):
            if fshed:
                predicted_scores[i] = pad_scores
        total_scores = predicted_scores + probs[:, None]
        linearized_scores = total_scores.view(-1)
        sorted_scores, sorted_ids = linearized_scores.sort(dim=-1, descending=True)
        # all 0 for now
        seq_ids = sorted_ids // total_scores.shape[-1]
        s_word_ids = sorted_ids % total_scores.shape[-1]
        # just taking first and only element for now
        probs = sorted_scores[:1]
        # beam size of 1
        beam_ids = seq_ids[:1]
        word_ids = s_word_ids[:1]
        words = [gpt2_tokenizer.decode(word_ids)]
        # add next token
        y = torch.cat([y[beam_ids], word_ids[:, None]], dim=1)
        # find out whether sequence is finished; currently only one sequence
        pre_finished = [finished[b_id.item()] for b_id in beam_ids]
        new_finished = [w_id.item() == gpt2_tokenizer.eos_token_id for w_id in word_ids]
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
    default="/checkpoint/rebeccaqian/backtranslation/2020-09-02/encoder/ep==787/",
    type=str,
    help="path to binarized model and config of saved encoder",
)
parser.add_argument(
    "--pretrained_decoder_path",
    default="/checkpoint/rebeccaqian/backtranslation/2020-09-02/decoder/ep==787/",
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
