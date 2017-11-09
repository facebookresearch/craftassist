#!/bin/bash
set -e


if [ $# == 1 ]; then
    IMAGE_TAG=$1
else
    echo "Usage: $0 <sha1>"
    exit 1
fi

MANIFEST=$(aws ecr batch-get-image \
    --repository-name craftassist \
    --region us-west-1 \
    --image-ids imageTag=$IMAGE_TAG \
    --query images[].imageManifest \
    --output text)

echo $MANIFEST

aws ecr put-image \
    --repository-name craftassist \
    --image-tag latest \
    --image-manifest "$MANIFEST"

echo
echo Success!
