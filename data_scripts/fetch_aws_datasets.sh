#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

ROOTDIR=$(dirname $(dirname $(realpath $0)))/
DATA_DIRNAME=datasets_folder
AGENT = $1

cd $ROOTDIR

echo "====== Downloading datasets to $ROOTDIR$DATA_DIRNAME.tar.gz ======"
curl http://craftassist.s3-us-west-2.amazonaws.com/pubr/$DATA_DIRNAME.tar.gz -o $DATA_DIRNAME.tar.gz

if [ -d "python/${AGENT}/datasets" ]
then
	echo "Overwriting datasets directory"
	rm -r python/$AGENT/datasets/*
else
	mkdir -p python/$AGENT/datasets/
fi

tar -xzvf $DATA_DIRNAME.tar.gz -C python/$AGENT/datasets/ --strip-components 1