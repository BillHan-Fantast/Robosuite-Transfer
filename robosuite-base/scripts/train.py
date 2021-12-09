import os
import numpy as np


import json
import torch
import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger

from util.rlkit_utils import experiment
import argparse

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Objective function
def run_experiment():

    # Load default variants
    try:
        with open(args.variant) as f:
            variant = json.load(f)
    except FileNotFoundError:
        print("Error opening specified variant json at: {}. "
              "Please check filepath and try again.".format(variant))

    # Change env configs

    variant["environment_kwargs"]["env_name"] = args.env
    variant["environment_kwargs"]["robots"] = args.robot
    variant["environment_kwargs"]["controller"] = args.controller

    # Set logging
    tmp_file_prefix = "{}_{}_{}".format(variant["environment_kwargs"]["env_name"],
                                        "".join(variant["environment_kwargs"]["robots"]),
                                        variant["environment_kwargs"]["controller"])
    if args.log_dir is not None:
        tmp_file_prefix = args.log_dir

    tmp_exp_prefix = "p{}_t{}_s{}".format(args.policy_lr, args.target_update, args.img_size)
    tmp_file_prefix = os.path.join(tmp_file_prefix, tmp_exp_prefix)

    # Change trainer configs
    variant["image_size"] = args.img_size
    variant["trainer_kwargs"]["policy_lr"] = args.policy_lr
    variant["trainer_kwargs"]["critic_target_update_frequency"] = args.target_update

    # Set agent
    agent = variant["algorithm"]

    # Setup logger
    tmp_dir = setup_logger(tmp_file_prefix, variant=variant, base_log_dir=variant["log_dir"], seed=args.seed)
    ptu.set_gpu_mode(torch.cuda.is_available())

    # Run experiment
    experiment(variant, agent=agent)


if __name__ == '__main__':

    # First, parse args
    # Define parser
    parser = argparse.ArgumentParser(description='RL and Env args')

    # Add seed arg always
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--variant', type=str, help='load algs an envs variant')
    parser.add_argument('--log-dir', type=str, help='basic log dir')
    parser.add_argument('--env', type=str, default='Lift', help='environment name')
    parser.add_argument('--robot', type=str, default='Panda', help='robot type')
    parser.add_argument('--controller', type=str, default='OSC_POSE', help='controller type')
    parser.add_argument('--img-size', type=int, default=128, help='image observation size')
    parser.add_argument('--policy-lr', type=float, default=0.001, help='policy learning rate')
    parser.add_argument('--target-update', type=int, default=5, help='target network update frequency of critics')
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Notify user we're starting run
    print('------------- Running --------------')

    print('Using variant: {}'.format(args.variant))

    # Execute run
    run_experiment()

    print('Finished run!')

