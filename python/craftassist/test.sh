#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.



cd $(dirname $0)
python3 -m unittest discover
