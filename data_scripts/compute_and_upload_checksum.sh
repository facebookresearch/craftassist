#!/bin/bash

# This script computes hashes for local directories and uploads them to AWS. Used to store hashes for the deployed models.

if [ -z $1 ]
then
    AGENT="craftassist"
    echo "Agent name not specified, defaulting to craftassist"
else
	AGENT=$1
fi

cd python/$AGENT
MODEL_CHECKSUM_PATH="models/checksum.txt"
DATA_CHECKSUM_PATH="datasets/checksum.txt"

# Compute hashes for local directories
echo "Computing hashes for python/${AGENT}/models/ and python/${AGENT}/datasets/"
find models/ -type f ! -name '*checksum*' -not -path '*/\.*' -print0 | sort -z | xargs -0 sha1sum | sha1sum > $MODEL_CHECKSUM_PATH
echo $MODEL_CHECKSUM_PATH
cat $MODEL_CHECKSUM_PATH
find datasets/ -type f ! -name '*checksum*' -not -path '*/\.*' -print0 | sort -z | xargs -0 sha1sum | sha1sum > $DATA_CHECKSUM_PATH 
echo $DATA_CHECKSUM_PATH
cat $DATA_CHECKSUM_PATH

aws s3 cp $MODEL_CHECKSUM_PATH s3://craftassist/pubr/model_checksum.txt
aws s3 cp $DATA_CHECKSUM_PATH s3://craftassist/pubr/data_checksum.txt
