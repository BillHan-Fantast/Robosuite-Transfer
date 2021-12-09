#!/bin/bash

export PYTHONPATH=$HOME/Desktop/Robot-Transfer/robosuite-conservative

env=$1
alg=$2
robot=$3
data=$4
seed=$5

declare -a beta=(1.0 4.0 10.0 20.0)

for t in 2 5
do
  for i in 1 2 3
  do
    CUDA_VISIBLE_DEVICES=${i} python offline_train.py --seed=${seed} \
        --alg-variant=configs/algs/${alg}.json \
        --env-variant=configs/envs/${robot}/${env}.json \
        --data-variant=configs/datasets/${data}.json \
        --beta=${beta[${i}]} \
        --target=${t} &
  done
done

wait

echo "All Done"