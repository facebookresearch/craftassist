#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.


CRAFTASSIST_PATH=$HOME/minecraft/python/craftassist/

if [ -z $1 ]
then
	MODELS_DIRNAME=models_folder
else
	MODELS_DIRNAME=$1
fi

echo "====== Compressing models directory ${CRAFTASSIST_PATH}models/ ======"

cd $CRAFTASSIST_PATH

tar -czvf $MODELS_DIRNAME.tar.gz models/

echo "tar file created at ${CRAFTASSIST_PATH}$MODELS_DIRNAME.tar.gz"

echo "====== Uploading models and datasets to S3 ======"
aws s3 cp ${CRAFTASSIST_PATH}$MODELS_DIRNAME.tar.gz s3://craftassist/pubr/
