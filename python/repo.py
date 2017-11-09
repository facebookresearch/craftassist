"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import os

python_dir = os.path.dirname(os.path.realpath(__file__))
repo_home = os.path.join(python_dir, "..")


def path(p):
    return os.path.join(repo_home, p)
