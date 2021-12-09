import os
import random

import numpy as np
import json
import torch
import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger

from experiment.offline_experiment import offline_img_cql_experiment, \
    offline_latent_experiment, offline_cql_experiment, deep_update_dict
import argparse

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Objective function
def run_experiment():

    # Load default variants
    with open(args.alg_variant) as f:
        alg_variant = json.load(f)
    with open(os.path.join(args.env_variant)) as f:
        env_variant = json.load(f)
    with open(os.path.join(args.data_variant)) as f:
        data_variant = json.load(f)
    variant = deep_update_dict(env_variant, alg_variant)
    variant = deep_update_dict(data_variant, variant)

    # Change configs
    variant["dataset_kwargs"]["max_epi_length"] = args.max_epi_len
    variant["trainer_kwargs"]["beta_penalty"] = args.beta
    variant["trainer_kwargs"]["target_update_frequency"] = args.target

    # Set logging
    tmp_file_prefix = "{}_{}_{}".format(variant["exp_name"],
                                        "".join(variant["environment_kwargs"]["robots"]),
                                        variant["dataset_kwargs"]["dataset_type"])
    exp_prefix = "{}_b{}_t{}".format(variant['algorithm'], args.beta, args.target)
    tmp_file_prefix = os.path.join(tmp_file_prefix, exp_prefix)

    variant["dataset_kwargs"]["dataset_path"] = os.path.join(variant["dataset_kwargs"]["dataset_path"],
                                                             variant["dataset_kwargs"]["dataset_type"] + ".hdf5")

    if args.log_dir is not None:
        tmp_file_prefix = args.log_dir

    # Setup logger
    tmp_dir = setup_logger(tmp_file_prefix, variant=variant, base_log_dir=variant["log_dir"],
                           tabular_log_files=['transfer.csv', 'model.csv', 'agent.csv'], seed=args.seed)
    ptu.set_gpu_mode(torch.cuda.is_available())

    if variant['algorithm'] in ['Latent_CQL', 'Latent_COMBO']:
        offline_latent_experiment(variant)
    elif variant['algorithm'] in ['IMG_CQL']:
        offline_img_cql_experiment(variant)
    elif variant['algorithm'] in ['CQL']:
        offline_cql_experiment(variant)
    else:
        raise NotImplementedError


if __name__ == '__main__':

    # First, parse args
    # Define parser
    parser = argparse.ArgumentParser(description='RL and Env args')

    # Add seed arg always
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--alg-variant', type=str, help='alg variant')
    parser.add_argument('--env-variant', type=str, default='Lift', help='env variant')
    parser.add_argument('--data-variant', type=str, help='dataset variant')
    parser.add_argument('--log-dir', type=str, help='basic log dir')
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--target', type=float, default=5.0)
    parser.add_argument('--max-epi-len', type=int, default=1000)
    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False  # reproducibility

    # Notify user we're starting run
    print('------------- Running --------------')

    # Execute run
    run_experiment()

    print('Finished run!')

