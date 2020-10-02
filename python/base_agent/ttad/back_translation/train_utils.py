import torch
from dataset import *
from modeling_gpt2 import *
from transformer import *


def compute_accuracy(outputs, y, tokenizer):
    """Compute model accuracy given predictions and targets. Used in validation.
    """
    # Do not include [CLS] token
    target_tokens = y[:, 1:]
    predicted_tokens = outputs.max(dim=-1)[1][:, :-1]
    acc = (predicted_tokens == target_tokens).sum(dim=1) == target_tokens.shape[-1]
    return acc


def validate(model, dataset, tokenizer):
    valid_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    tot_acc = 0.0
    tot_loss = 0.0
    steps = 0
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            trees, text = batch
            text_idx_ls = [tokenizer.encode(cmd) for cmd in text]
            tree_idx_ls = [tokenizer.encode(tree) for tree in trees]
            x, x_mask, y, y_mask = collate(tree_idx_ls, text_idx_ls, tokenizer)
            out = model(x, x_mask, y, y_mask)
            loss, predictions = out[:2]
            lm_acc = compute_accuracy(predictions, y, tokenizer)
            acc = lm_acc.sum().item() / lm_acc.shape[0]
            tot_acc += acc
            tot_loss += loss.item()
            steps += 1
    print("Valid accuracy: {} Loss: {}".format(tot_acc / steps, tot_loss / steps))
