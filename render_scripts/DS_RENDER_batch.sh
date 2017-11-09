#!/bin/bash -l

module purge
module load java

for i in `seq $1 $2`; do
  op=gather/${i}
  mkdir -p ${op}
  echo ${i}
  python ../python/render_for_dataset.py  --out-dir=${op} --seed=$i
  # echo "python python/render_for_dataset.py  --out-dir=${op} --seed=$i > check.log"
done
