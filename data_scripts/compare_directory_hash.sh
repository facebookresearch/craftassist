#!/bin/bash

# This script checks if models and datasets are up to date, and either triggers a download or gives the user a warning to update local files.
if [ -z $1 ]
then
    AGENT="craftassist"
    echo "Agent name not specified, defaulting to craftassist"
else
    AGENT=$1
fi
# Compute hashes for local directories
echo "Computing hashes for python/${AGENT}/models/ and python/${AGENT}/datasets/"
cd python/$AGENT
find models/ -type f ! -name '*checksum*' -not -path '*/\.*' -print0 | sort -z | xargs -0 sha1sum | sha1sum > models/checksum.txt
find datasets/ -type f ! -name '*checksum*' -not -path '*/\.*' -print0 | sort -z | xargs -0 sha1sum | sha1sum > datasets/checksum.txt
cd ../..

echo
# Download AWS checksum
echo "Downloading latest hash from AWS"
curl http://craftassist.s3-us-west-2.amazonaws.com/pubr/model_checksum.txt -o python/$AGENT/models/model_checksum_aws.txt
echo "Latest model hash in python/$AGENT/models/model_checksum_aws.txt"
curl http://craftassist.s3-us-west-2.amazonaws.com/pubr/data_checksum.txt -o python/$AGENT/datasets/data_checksum_aws.txt
echo "Latest data hash in python/$AGENT/datasets/data_checksum_aws.txt"
echo

if cmp -s python/$AGENT/models/checksum.txt python/$AGENT/models/model_checksum_aws.txt
then
	echo "Local models directory is up to date."
else
	echo "Local models directory is out of sync. Would you like to download the updated files from AWS? This overwrites python/${AGENT}/models/"
	read -p "Enter Y/N: " permission
	echo $permission
	if [ "$permission" == "Y" ] || [ "$permission" == "y" ] || [ "$permission" == "yes" ]; then
		echo "Downloading models directory"
		SCRIPT_PATH="$PWD/data_scripts/fetch_aws_models.sh"
		echo $SCRIPT_PATH
		. "$SCRIPT_PATH" "$AGENT"
	else
		echo "Warning: Outdated models can cause breakages in the repo."
	fi
fi

if cmp -s python/$AGENT/datasets/checksum.txt python/$AGENT/datasets/data_checksum_aws.txt
then
        echo "Local datasets directory is up to date."
else
        echo "Local datasets directory is out of sync. Would you like to download the updated files from AWS? This overwrites python/${AGENT}/datasets/"
        read -p "Enter Y/N: " permission
        echo $permission
        if [ "$permission" == "Y" ] || [ "$permission" == "y" ] || [ "$permission" == "yes" ]; then
                echo "Downloading datasets directory"
                SCRIPT_PATH="$PWD/data_scripts/fetch_aws_datasets.sh"
                echo $SCRIPT_PATH
                . "$SCRIPT_PATH" "$AGENT"
                exit 1
        else
                echo "Warning: Outdated datasets can cause breakages in the repo."
                exit 1
        fi
fi
