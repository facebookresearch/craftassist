"""
Copyright (c) Facebook, Inc. and its affiliates.

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
        Noop.CHATS += [
            x
            for x in chats
            if x
            not in [
                "good job",
                "cool",
                "that is really cool",
                "that is awesome",
                "awesome",
                "that is amazing",
                "that looks good",
                "you did well",
                "great",
                "good",
                "nice",
                "that is wrong",
                "that was wrong",
                "that was completely wrong",
                "not that",
                "that looks horrible",
                "that is not what i asked",
                "that is not what i told you to do",
                "that is not what i asked for",
                "not what i told you to do",
                "you failed",
                "failure",
                "fail",
                "not what i asked for",
                "where are you",
                "tell me where you are",
                "i do n't see you",
                "i ca n't find you",
                "are you still around",
                "what are you doing",
                "tell me what are you doing",
                "what is your task",
                "tell me your task",
                "what are you up to",
                "stop",
                "wait",
                "where are you going",
                "what is this",
                "come here",
                "mine",
                "what is that thing",
                "come back",
                "go back",
                "what is that",
                "keep going",
                "tower",
                "follow me",
                "do n't do that",
                "do n't move",
                "hold on",
                "this is pretty",
                "continue",
                "can you follow me",
                "move",
                "this is nice",
                "this is sharp",
                "this is very big",
                "keep digging",
                "circle",
                "that is sharp",
                "it looks nice",
            ]
        ]
    except:
        print("chats file not found")

    random.seed(args.seed)
    for text, d in zip(*generate_actions(args.n, args.action_type)):
        for sentence in text:
            print(json.dumps(sentence))
        print(json.dumps(d))
        print()
