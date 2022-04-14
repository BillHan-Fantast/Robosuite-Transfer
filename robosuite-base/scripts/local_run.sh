#!/bin/bash

export PYTHONPATH=$HOME/Desktop/Robot-Transfer/robosuite-base

CUDA_VISIBLE_DEVICES=0 python scripts/train.py --seed=0 \
    --variant=configs/frontview.json \
    --env=Reach \
    --robot=Panda \
    --controller=OSC_POSE \
    --img-size=84 \
    --target-update=5