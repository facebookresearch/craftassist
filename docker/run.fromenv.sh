#!/bin/bash
set -u

echo $RUN_SH_GZ_B64 | base64 --decode | gunzip > /run.sh
chmod +x /run.sh
/run.sh
