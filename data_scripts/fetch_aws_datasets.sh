#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

ROOTDIR=$(dirname $(dirname $(realpath $0)))/
if [ -z $1 ]
then
	MODELS_DIRNAME=models_folder
else
	MODELS_DIRNAME=$1
fi

if [ -z $2 ]
then
	GROUND_TRUTH_FNAME=ground_truth_data
else
	GROUND_TRUTH_FNAME=$2
fi

cd $ROOTDIR

echo "====== Downloading models and datasets to $ROOTDIR$MODELS_DIRNAME.tar.gz ======"

curl http://craftassist.s3-us-west-2.amazonaws.com/pubr/$MODELS_DIRNAME.tar.gz -o $MODELS_DIRNAME.tar.gz 
tar -xzvf $MODELS_DIRNAME.tar.gz -C python/craftassist/models/ --strip-components 1

echo "====== Downloading ground truth data to ${ROOTDIR}python/craftassist/$GROUND_TRUTH_FNAME.txt ======"

curl http://craftassist.s3-us-west-2.amazonaws.com/pubr/$GROUND_TRUTH_FNAME.txt -o python/craftassist/$GROUND_TRUTH_FNAME.txt
