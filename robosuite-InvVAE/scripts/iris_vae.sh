#!/bin/bash

export PYTHONPATH=/iris/u/beining/robot-transfer/robosuite-InvVAE

alg=$1
env=$2
pair=$3
repr=$4
domain=$5
lr=$6
beta=$7
inverse=$8
paired=$9
seed=${10}

for b in 0.1 0.4
do
    python vae_runs/offline_vae.py --seed=${seed} \
        --alg-variant=configs/algs/${alg}.json \
        --env-variant=configs/envs/${env}.json \
        --wrapper-variant=configs/wrappers/frontview.json \
        --sweep-variant=configs/algs/transfer_sweep.json \
        --paired-variant=configs/envs/Paired/${pair}.json \
        --repr-size=${repr} \
        --beta=${b} \
        --lr=${lr} \
        --domain=${domain} \
        --inverse=${inverse} \
        --paired=${paired} &
done

wait

echo "All Done"