#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.



cd $(dirname $0)/../python/craftassist
mypy --ignore-missing-imports craftassist_agent.py
