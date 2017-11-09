import argparse
import json
import os
import sys

stack_agent_dir = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(stack_agent_dir)

from preprocess import preprocess_chat
from ttad.train_ttad_model import ActionDictBuilder

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="../../models/ttad.pth", help="model to test")
parser.add_argument(
    "--chat_file",
    default="new_chats.txt",
    help="The text file that contains the new chats to be annotated.",
)
args = parser.parse_args()

listener = ActionDictBuilder(args.model)

chats, processed_chats = [], []

with open(args.chat_file) as f:
    chats = [line.strip() for line in f.readlines()]
    processed_chats = [(x, preprocess_chat(x)[0]) for x in chats if len(x.strip()) > 0]

chat_dict = {}
for x, y in processed_chats:
    chat_dict[y] = chat_dict.get(y, []) + [x]

processed_chats = [(chat_lst, pro_chat) for pro_chat, chat_lst in chat_dict.items()]

res = []

start_at = 0
res = res[:start_at]
for i, (chat, proc_chat) in enumerate(processed_chats):
    if i >= start_at:
        acc, tree = listener.annotate(proc_chat)
        res += [(chat, proc_chat, acc, tree)]
        if i > 0 and i % 10 == 0:
            print(i)
            with open("chat_valid_%d.json" % (i,), "w") as f:
                json.dump(res, f)

with open("chat_valid_full.json", "w") as f:
    json.dump(res, f)
