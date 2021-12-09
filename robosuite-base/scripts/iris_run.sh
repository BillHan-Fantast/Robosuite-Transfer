#!/bin/bash

export PYTHONPATH=/iris/u/beining/robot-transfer/robosuite-base

env=$1
robot=$2
ctrl=$3
var=$4
pl=$5

for t in 5 10
do
  for i in 84 128
  do
    for s in 1 2 3
    do
      python scripts/train.py --seed=$s \
      --variant=configs/${var}.json \
      --env=${env} \
      --robot=${robot} \
      --controller=${ctrl} \
      --policy-lr=${pl} \
      --img-size=${i} \
      --target-update=${t} &
    done
  done
  wait
done

echo "All Done"