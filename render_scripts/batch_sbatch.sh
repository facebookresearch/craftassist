#!/bin/bash
for i in $(seq 1 500 12500)
do
  # echo "welcome $i $(($i+99))"
  # echo "sbatch SBATCH_job.sh $i $(($i+499))"
  sbatch SBATCH_job.sh $i $(($i+499))
done
