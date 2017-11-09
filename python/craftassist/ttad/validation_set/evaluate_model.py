import argparse
import json
import os
import sys

stack_agent_dir = os.path.dirname(os.getcwd())
sys.path.append(stack_agent_dir)

from train_ttad_model import ActionDictBuilder

"""Compare two nested dictionaries d1 and d2.
Ignore the 'has_attribute_' key right now"""


def compare_dicts(d1, d2):
    for k in d1.keys():
        if k not in d2.keys():
            return False
        else:
            if type(d1[k]) is dict:
                x = compare_dicts(d1[k], d2[k])
                if not x:
                    return False
            else:
                if (d1[k] != d2[k]) and (k != "has_attribute_"):
                    return False
    return True


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="../../models/ttad.pth", help="model to test")
parser.add_argument(
    "--eval_file",
    default="validation_set_rephrase_only.json",
    help="The json file that contains validation set",
)
args = parser.parse_args()

listener = ActionDictBuilder(args.model)


original_data = []
data_ground_truth_mapping = {}

with open(args.eval_file) as f:
    original_data = json.load(f)

total_correct = 0
total = 0

for chat in original_data:
    total += 1
    chat_text = chat[0]
    gt_chat_dict = chat[1]
    # remove description from the original dict
    action = list(gt_chat_dict.keys())[0]
    gt_chat_dict[action].pop("description", None)

    predicted_chat_dict = listener.listen(chat_text)
    if compare_dicts(gt_chat_dict, predicted_chat_dict):
        total_correct += 1
    else:
        print("Text: %r" % (chat_text))
        print("Ground truth: %r" % (gt_chat_dict))
        print("Model prediction: %r" % (predicted_chat_dict))
        print("-" * 30)


print("Total correct percentage: %r" % (((total_correct * 1.0) / total) * 100))
