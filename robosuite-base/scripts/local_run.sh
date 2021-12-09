#!/bin/bash

export PYTHONPATH=$HOME/Desktop/Robot-Transfer/robosuite-base

env=$1
robot=$2
ctrl=$3
var=$4
pl=$5

for i in 84 128
do
  for s in 1 2 3
  do
    CUDA_VISIBLE_DEVICES=$s python scripts/train.py --seed=$s \
      --variant=configs/${var}.json \
      --env=${env} \
      --robot=${robot} \
      --controller=${ctrl} \
      --policy-lr=${pl} \
      --img-size=${i} \
      --target-update=5 &
  done
done
wait

echo "All Done"