"""
This file generates action trees and language based on options from
command line.
"""
import json
from generate_data import *


class Action(ActionNode):
    """options for Actions"""

    CHOICES = [
        Move,
        Build,
        Destroy,
        Noop,
        Stop,
        Resume,
        Dig,
        Copy,
        Undo,
        Fill,
        Spawn,
        Freebuild,
        Dance,
        GetMemory,
        PutMemory,
    ]


# Mapping of command line action type to action class
action_type_map = {
    "move": Move,
    "build": Build,
    "destroy": Destroy,
    "noop": Noop,
    "stop": Stop,
    "resume": Resume,
    "dig": Dig,
    "copy": Copy,
    "undo": Undo,
    "fill": Fill,
    "spawn": Spawn,
    "freebuild": Freebuild,
    "dance": Dance,
    "get_memory": GetMemory,
    "put_memory": PutMemory,
}


def generate_actions(n, action_type=None):
    """ Generate action tree and language based on action type """
    texts = []
    dicts = []
    for _ in range(n):
        action_name = action_type if type(action_type) is list else action_type_map[action_type]
        a = Action.generate(action_name)
        texts.append(a.generate_description())
        dicts.append(a.to_dict())
    return texts, dicts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=100)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--chats_file", "-chats", type=str, default="noop_dataset.txt")
    parser.add_argument("--action_type", default=Action.CHOICES)
    args = parser.parse_args()

    # load file containing negative examples of chats
    try:
        f = open(args.chats_file)
        chats = [line.strip() for line in f]
        f.close()
        Noop.CHATS += chats
    except:
        print("chats file not found")

    random.seed(args.seed)
    for text, d in zip(*generate_actions(args.n, args.action_type)):
        for sentence in text:
            print(json.dumps(sentence))
        print(json.dumps(d))
        print()
