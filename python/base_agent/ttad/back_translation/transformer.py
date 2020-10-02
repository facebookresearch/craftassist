from transformers.modeling_bert import BertModel
import torch.nn as nn


class TransformerEncoder(nn.Module):
    """Transformer Encoder class.
    """

    def __init__(self, config, tokenizer):
        super(TransformerEncoder, self).__init__()
        # Initializes transformer architecture with BERT structure
        self.bert = BertModel(config)

    def forward(self, input_ids, attention_mask):
        bert_model = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_out = bert_model[0]
        return hidden_out


class EncoderDecoder(nn.Module):
    """Encoder decoder architecture used for back translation.
    """

    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, x_mask, y, y_mask, labels, use_lm=False):
        hidden_out = self.encoder(input_ids=x, attention_mask=x_mask)[0]
        dec_out = self.decoder(
            input_ids=y,
            attention_mask=y_mask,
            encoder_hidden_states=hidden_out,
            encoder_attention_mask=x_mask,
            labels=labels,
        )
        return dec_out

    def step(self, y, x_reps):
        """Used during beam search.
        NOTE: This is a placeholder.
        """
        dec_out = self.decoder(input_ids=y, encoder_hidden_states=x_reps)
        return dec_out
