"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import fileinput

from ttad_annotate import MAX_WORDS

print("command", *["word{}".format(i) for i in range(MAX_WORDS)], sep=",")

for line in fileinput.input():
    command = line.replace(",", "").strip()
    words = command.split()
    print(command, *words, *([""] * (MAX_WORDS - len(words))), sep=",")
