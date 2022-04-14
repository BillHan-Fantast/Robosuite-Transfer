#!/bin/bash

export PYTHONPATH=/iris/u/beining/robot-transfer/robosuite-InvVAE

python vae_runs/offline_vae.py --seed=0 \
        --alg-variant=configs/algs/vae_cql.json \
        --env-variant=configs/envs/Sawyer_to_IIWA/Reach_light_MR.json \
        --wrapper-variant=configs/wrappers/frontview.json \
        --sweep-variant=configs/algs/transfer_sweep.json \
        --paired-variant=configs/envs/Paired/Reach_2k.json \
        --repr-size=20 \
        --beta=0.1 \
        --lr=0.0001 \
        --domain=10 \
        --inverse=10.0 \
        --paired=50.0