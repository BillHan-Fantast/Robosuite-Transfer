#!/bin/bash

export PYTHONPATH=/iris/u/beining/robot-transfer/robosuite-conservative

env=$1
alg=$2
robot=$3
data=$4

for s in 0
do
  for b in 1.0 4.0 10.0 20.0
  do
    for t in 2 5
    do
      python offline_train.py --seed=${s} \
        --alg-variant=configs/algs/${alg}.json \
        --env-variant=configs/envs/${robot}/${env}.json \
        --data-variant=configs/datasets/${data}.json \
        --beta=${b} \
        --target=${t} &
    done
  done
done

wait

echo "All Done"