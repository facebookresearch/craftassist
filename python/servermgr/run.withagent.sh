#!/bin/bash -x
# Copyright (c) Facebook, Inc. and its affiliates.



S3_DEST=s3://craftassist/workdirs

function background_agent() (
    python3 /minecraft/python/wait_for_cuberite.py --host localhost --port 25565
    python3 /minecraft/python/craftassist/craftassist_agent.py 1>agent.log 2>agent.log
)
background_agent &

python3 /minecraft/python/cuberite_process.py \
    --mode creative \
    --workdir . \
    --config diverse_world \
    --seed 0 \
    --logging \
    --add-plugin shutdown_on_leave \
    1>cuberite_process.log \
    2>cuberite_process.log


TARBALL=workdir.$(date '+%Y-%m-%d-%H:%M:%S').$(hostname).tar.gz
tar czf $TARBALL . --force-local

if [ -z "$CRAFTASSIST_NO_UPLOAD" ]; then
    # expects $AWS_ACCESS_KEY_ID and $AWS_SECRET_ACCESS_KEY to exist
    aws s3 cp $TARBALL $S3_DEST/$TARBALL
fi

halt
