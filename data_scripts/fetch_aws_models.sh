#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

ROOTDIR=$(dirname $(dirname $(realpath $0)))/
MODELS_DIRNAME=models_folder
AGENT = $1

cd $ROOTDIR

if [ -d "python/${AGENT}/models" ]
then
        echo "Overwriting models directory"
        rm -r python/$AGENT/models/*
else
        mkdir -p python/$AGENT/models/
fi

echo "====== Downloading models to $ROOTDIR$MODELS_DIRNAME.tar.gz ======"

curl http://craftassist.s3-us-west-2.amazonaws.com/pubr/$MODELS_DIRNAME.tar.gz -o $MODELS_DIRNAME.tar.gz
tar -xzvf $MODELS_DIRNAME.tar.gz -C python/$AGENT/models/ --strip-components 1