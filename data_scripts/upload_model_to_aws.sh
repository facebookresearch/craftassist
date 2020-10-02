#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.


CRAFTASSIST_PATH=$PWD/python/craftassist/

if [ -z $1 ]
then
	MODELS_DIRNAME=models_folder
else
	MODELS_DIRNAME=$1
fi

DATA_DIRNAME=datasets_folder

echo "====== Compressing models directory ${CRAFTASSIST_PATH}models/ ======"

cd $CRAFTASSIST_PATH

tar -czvf $MODELS_DIRNAME.tar.gz models/

echo "tar file created at ${CRAFTASSIST_PATH}$MODELS_DIRNAME.tar.gz"

echo "====== Uploading models and datasets to S3 ======"
aws s3 cp ${CRAFTASSIST_PATH}$MODELS_DIRNAME.tar.gz s3://craftassist/pubr/

echo "====== Compressing data directory ${CRAFTASSIST_PATH}datasets/ ======"

tar -czvf $DATA_DIRNAME.tar.gz datasets/

echo "tar file created at ${CRAFTASSIST_PATH}$DATA_DIRNAME.tar.gz"

echo "====== Uploading datasets to S3 ======"
aws s3 cp ${CRAFTASSIST_PATH}$DATA_DIRNAME.tar.gz s3://craftassist/pubr/
