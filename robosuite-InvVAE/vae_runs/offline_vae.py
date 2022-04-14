import os
import random
import argparse

import numpy as np
import json
import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger
from experiment.offline_experiment import offline_vae_cql_experiment, \
                                          offline_vae_sharing_experiment, \
                                          deep_update_dict

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_sweep_configs(kwargs):
    keys = list(kwargs.keys())
    num_keys = len(keys)
    num_configs = 1
    num_range_lens = []
    for key in keys:
        num_configs *= len(kwargs[key])
        num_range_lens.append(len(kwargs[key]))

    num_range_lens.append(1)
    residual_epoch, iterate_epoch = num_range_lens[:-1], []
    for i in range(num_keys):
        iterate_epoch.append(int(np.prod(num_range_lens[i+1:])))

    configs = []
    for i in range(num_configs):
        config = {}
        for j in range(num_keys):
            residual = int(i // iterate_epoch[j])
            residual = residual % residual_epoch[j]
            config[keys[j]] = kwargs[keys[j]][residual]
        configs.append(config)

    return configs


def vae_experiment():

    # Load default variants
    with open(args.alg_variant) as f:
        alg_variant = json.load(f)
    with open(os.path.join(args.env_variant)) as f:
        env_variant = json.load(f)
    with open(os.path.join(args.wrapper_variant)) as f:
        wrapper_variant = json.load(f)
    with open(os.path.join(args.paired_variant)) as f:
        paired_variant = json.load(f)
    variant = deep_update_dict(env_variant, alg_variant)
    variant = deep_update_dict(wrapper_variant, variant)
    variant = deep_update_dict(paired_variant, variant)

    # Change configs
    variant["algorithm_kwargs"]["save_snapshot"] = bool(args.checkpoint)
    variant["algorithm_kwargs"]["save_interval"] = args.save_interval
    variant["vae_model_kwargs"]["representation_size"] = args.repr_size
    variant["transfer_trainer_kwargs"]["vae_lr"] = args.lr
    variant["transfer_trainer_kwargs"]["beta"] = args.beta
    variant["transfer_trainer_kwargs"]["c_domain"] = args.domain
    variant["transfer_trainer_kwargs"]["c_paired"] = args.paired
    variant["transfer_trainer_kwargs"]["c_src_inverse"] = args.inverse
    variant["transfer_trainer_kwargs"]["c_trg_inverse"] = args.inverse

    # Set logging
    tmp_file_prefix = "{}_{}_InvVAE_Transfer".format(variant["exp_name"], variant["paired_name"])
    exp_prefix = "transfer_r{}_lr{}_b{}_d{}_i{}_p{}".format(args.repr_size, args.lr, args.beta,
                                                            args.domain, args.inverse, args.paired)
    tmp_file_prefix = os.path.join(tmp_file_prefix, exp_prefix)

    if args.log_dir is not None:
        tmp_file_prefix = args.log_dir

    ptu.set_gpu_mode(torch.cuda.is_available())

    # Setup logger
    log_dir, checkpoint = setup_logger(tmp_file_prefix, variant=variant, base_log_dir=variant["log_dir"],
                                       tabular_log_files=['agent.csv', 'model.csv', 'transfer.csv'],
                                       checkpoint=bool(args.checkpoint), seed=args.seed)

    variant['checkpoint_dir'] = log_dir
    variant['load_checkpoint'] = checkpoint
    return offline_vae_cql_experiment(variant)


def sharing_experiment(model_path, sweep_kwargs):

    # Load default variants
    configs = get_sweep_configs(sweep_kwargs)

    for config in configs:
        with open(args.alg_variant) as f:
            alg_variant = json.load(f)
        with open(os.path.join(args.env_variant)) as f:
            env_variant = json.load(f)
        with open(os.path.join(args.wrapper_variant)) as f:
            wrapper_variant = json.load(f)
        variant = deep_update_dict(env_variant, alg_variant)
        variant = deep_update_dict(wrapper_variant, variant)

        # Change configs
        tmp_file_prefix = model_path.split('/')
        tmp_file_prefix = os.path.join(*tmp_file_prefix[1:-2])
        tmp_file_prefix = tmp_file_prefix.replace('transfer', 'sharing')
        variant["algorithm_kwargs"]["save_snapshot"] = bool(args.checkpoint)
        variant["algorithm_kwargs"]["save_interval"] = args.save_interval
        variant["algorithm_kwargs"]["transfer_model_path"] = model_path
        for k, v in config.items():
            if k == 'beta_penalty':
                variant["policy_trainer_kwargs"]["beta_penalty"] = v
                tmp_file_prefix += '_beta' + str(v)
            elif k == 'lambda_penalty':
                variant["algorithm_kwargs"]["lambda_penalty"] = v
                tmp_file_prefix += '_lam' + str(v)
            else:
                raise NotImplementedError

        if args.log_dir is not None:
            tmp_file_prefix = args.log_dir

        ptu.set_gpu_mode(torch.cuda.is_available())

        # Setup logger
        log_dir, checkpoint = setup_logger(tmp_file_prefix, variant=variant, base_log_dir=variant["log_dir"],
                                           tabular_log_files=['agent.csv'], checkpoint=bool(args.checkpoint),
                                           seed=args.seed)

        variant['checkpoint_dir'] = log_dir
        variant['load_checkpoint'] = checkpoint
        offline_vae_sharing_experiment(variant)


if __name__ == '__main__':

    # First, parse args
    parser = argparse.ArgumentParser(description='RL and Env args')

    # Add seed arg always
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--alg-variant', type=str, help='alg variant')
    parser.add_argument('--env-variant', type=str, help='env variant')
    parser.add_argument('--wrapper-variant', type=str, help='wrapper variant')
    parser.add_argument('--paired-variant', type=str, help='paired dataset')
    parser.add_argument('--sweep-variant', type=str, help='sweep variant')
    parser.add_argument('--log-dir', type=str, help='basic log dir')
    parser.add_argument('--checkpoint', type=int, default=0)
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--repr-size', type=int, default=20)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--domain', type=float, default=10.0)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--paired', type=float, default=0.0)
    parser.add_argument('--inverse', type=float, default=0.0)

    # Training parameters
    args = parser.parse_args()

    # Set random seed and Disable randomness
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Notify user we're starting run
    print('------------- Running --------------')

    # Execute run
    with open(args.sweep_variant) as f:
        sweep_variant = json.load(f)
    model_path = vae_experiment()
    sharing_experiment(model_path, sweep_variant)

    print('Finished run!')



