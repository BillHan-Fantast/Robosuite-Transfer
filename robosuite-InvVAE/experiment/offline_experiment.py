
import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import ActionNormalizedEnv
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.samplers.rollout_functions import robo_rollout
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.policies import TanhGaussianPolicy

from models.dataset import TrjRoboDataset, StepRoboDataset
from models.cql.cql import CQLTrainer
from models.cql.policies import VAEMakeDeterministic

from models.vae.conv_vae import InvVAE
from models.vae.vae_trainer import ConvVAETrainer
from models.vae.replay_dataset import RobotReplayDataset, RobotSharingReplayDataset
from models.dynamics.dynamics_trainer import DynTrainer

from experiment.offline_algorithm import OfflineInvVAETransferAlgorithm, OfflineInvVAESharingAlgorithm

import robosuite as suite
from robosuite.wrappers import GymVisualWrapper
from robosuite.controllers import load_controller_config

import torch
import os.path as osp


def offline_vae_cql_experiment(variant):
    source_env_config = variant["source_environment_kwargs"]
    target_env_config = variant["target_environment_kwargs"]
    source_wrapper_config = variant["source_wrapper_kwargs"]
    target_wrapper_config = variant["target_wrapper_kwargs"]

    source_controller = source_env_config.pop("controller")
    target_controller = target_env_config.pop("controller")
    source_controller_config = load_controller_config(default_controller=source_controller)
    target_controller_config = load_controller_config(default_controller=target_controller)

    source_env = suite.make(**source_env_config, has_renderer=False, controller_configs=source_controller_config)
    source_env = ActionNormalizedEnv(GymVisualWrapper(source_env, **source_wrapper_config))
    target_env = suite.make(**target_env_config, has_renderer=False, controller_configs=target_controller_config)
    target_env = ActionNormalizedEnv(GymVisualWrapper(target_env, **target_wrapper_config))

    # (H, W, C)cc
    image_obs_shape = source_env.observation_space['image_obses'].low.shape
    assert image_obs_shape == target_env.observation_space['image_obses'].low.shape
    robot_state_shape = source_env.observation_space['robot_states'].low.shape
    assert robot_state_shape == target_env.observation_space['robot_states'].low.shape
    action_shape = source_env.action_space.low.shape
    assert action_shape == target_env.action_space.low.shape

    use_state_input = variant["use_state_input"]
    vae_latent_dim = variant['vae_model_kwargs']['representation_size']
    action_dim = action_shape[0]
    q_input_dim = vae_latent_dim + action_dim
    policy_input_dim = vae_latent_dim
    if use_state_input:
        q_input_dim += robot_state_shape[0]
        policy_input_dim += robot_state_shape[0]

    src_qf1 = ConcatMlp(input_size=q_input_dim, output_size=1, **variant['qf_kwargs'])
    src_qf2 = ConcatMlp(input_size=q_input_dim, output_size=1, **variant['qf_kwargs'])
    src_target_qf1 = ConcatMlp(input_size=q_input_dim, output_size=1, **variant['qf_kwargs'])
    src_target_qf2 = ConcatMlp(input_size=q_input_dim, output_size=1, **variant['qf_kwargs'])

    trg_qf1 = ConcatMlp(input_size=q_input_dim, output_size=1, **variant['qf_kwargs'])
    trg_qf2 = ConcatMlp(input_size=q_input_dim, output_size=1, **variant['qf_kwargs'])
    trg_target_qf1 = ConcatMlp(input_size=q_input_dim, output_size=1, **variant['qf_kwargs'])
    trg_target_qf2 = ConcatMlp(input_size=q_input_dim, output_size=1, **variant['qf_kwargs'])

    # Instantiate trainer with appropriate agent
    src_agent = TanhGaussianPolicy(
            obs_dim=policy_input_dim,
            action_dim=action_dim,
            **variant['policy_kwargs']
    )
    trg_agent = TanhGaussianPolicy(
            obs_dim=policy_input_dim,
            action_dim=action_dim,
            **variant['policy_kwargs']
    )
    src_policy_trainer = CQLTrainer(
            action_shape=(action_dim,),
            policy=src_agent,
            qf1=src_qf1,
            qf2=src_qf2,
            target_qf1=src_target_qf1,
            target_qf2=src_target_qf2,
            use_state_input=use_state_input,
            **variant['policy_trainer_kwargs']
    )
    trg_policy_trainer = CQLTrainer(
            action_shape=(action_dim,),
            policy=trg_agent,
            qf1=trg_qf1,
            qf2=trg_qf2,
            target_qf1=trg_target_qf1,
            target_qf2=trg_target_qf2,
            use_state_input=use_state_input,
            **variant['policy_trainer_kwargs']
    )

    # Transfer Module and Trainer
    vae_model = InvVAE(
        image_shape=image_obs_shape,
        action_shape=action_shape,
        **variant['vae_model_kwargs']
    )
    transfer_trainer = ConvVAETrainer(
        model=vae_model,
        **variant['transfer_trainer_kwargs']
    )
    dynamics_trainer = DynTrainer(
        vae_model,
        **variant['dynamics_trainer_kwargs']
    )

    # Eval Agent, Pay Attention to the Agent Assignment
    eval_source_agent = VAEMakeDeterministic(trg_agent, vae_model, use_state_input, 'source')
    eval_target_agent = VAEMakeDeterministic(src_agent, vae_model, use_state_input, 'target')

    # Datasets and Buffers
    src_train_buffer = RobotReplayDataset(
        robot_state_shape[0], vae_latent_dim, action_shape[0],
        variant['algorithm_kwargs']['policy_buffer_size']
    )
    trg_train_buffer = RobotReplayDataset(
        robot_state_shape[0], vae_latent_dim, action_shape[0],
        variant['algorithm_kwargs']['policy_buffer_size']
    )
    eval_source_collector = MdpPathCollector(source_env, eval_source_agent, rollout_fn=robo_rollout)
    eval_target_collector = MdpPathCollector(target_env, eval_target_agent, rollout_fn=robo_rollout)
    src_general_dataset = StepRoboDataset(**variant['source_general_dataset_kwargs'])
    trg_general_dataset = StepRoboDataset(**variant['target_general_dataset_kwargs'])
    src_vae_dataset = StepRoboDataset(**variant['source_transfer_dataset_kwargs'])
    trg_vae_dataset = StepRoboDataset(**variant['target_transfer_dataset_kwargs'])
    paired_vae_dataset = StepRoboDataset(**variant['paired_general_dataset_kwargs'])
    src_eval_dataset = StepRoboDataset(**variant['source_eval_dataset_kwargs'])
    trg_eval_dataset = StepRoboDataset(**variant['target_eval_dataset_kwargs'])
    src_train_dataset = TrjRoboDataset(**variant['source_train_dataset_kwargs'])
    trg_train_dataset = TrjRoboDataset(**variant['target_train_dataset_kwargs'])

    # Define algorithm
    algorithm = OfflineInvVAETransferAlgorithm(
        source_agent=src_agent,
        target_agent=trg_agent,
        transfer_model=vae_model,

        source_trainer=src_policy_trainer,
        target_trainer=trg_policy_trainer,
        transfer_trainer=transfer_trainer,
        dynamics_trainer=dynamics_trainer,

        eval_source_env=source_env,
        eval_target_env=target_env,
        eval_source_collector=eval_source_collector,
        eval_target_collector=eval_target_collector,

        source_general_dataset=src_general_dataset,
        target_general_dataset=trg_general_dataset,
        source_vae_dataset=src_vae_dataset,
        target_vae_dataset=trg_vae_dataset,
        paired_vae_dataset=paired_vae_dataset,
        source_eval_dataset=src_eval_dataset,
        target_eval_dataset=trg_eval_dataset,
        source_train_dataset=src_train_dataset,
        target_train_dataset=trg_train_dataset,
        source_train_buffer=src_train_buffer,
        target_train_buffer=trg_train_buffer,

        save_model_path=variant['checkpoint_dir'],
        **variant['algorithm_kwargs']
    )

    algorithm.to(ptu.device)
    if variant['algorithm_kwargs']['load_dynamics_model']:
        path = osp.join(variant['algorithm_kwargs']['dynamics_model_path'], 'dynamics_model.pkl')
        algorithm.load_dynamics_model(path)
    if variant['load_checkpoint']:
        algorithm.load_snapshot()
    return algorithm.train()


def offline_vae_sharing_experiment(variant):
    source_env_config = variant["source_environment_kwargs"]
    target_env_config = variant["target_environment_kwargs"]
    source_wrapper_config = variant["source_wrapper_kwargs"]
    target_wrapper_config = variant["target_wrapper_kwargs"]

    source_controller = source_env_config.pop("controller")
    target_controller = target_env_config.pop("controller")
    source_controller_config = load_controller_config(default_controller=source_controller)
    target_controller_config = load_controller_config(default_controller=target_controller)

    source_env = suite.make(**source_env_config, has_renderer=False, controller_configs=source_controller_config)
    source_env = ActionNormalizedEnv(GymVisualWrapper(source_env, **source_wrapper_config))
    target_env = suite.make(**target_env_config, has_renderer=False, controller_configs=target_controller_config)
    target_env = ActionNormalizedEnv(GymVisualWrapper(target_env, **target_wrapper_config))

    # (H, W, C)cc
    image_obs_shape = source_env.observation_space['image_obses'].low.shape
    assert image_obs_shape == target_env.observation_space['image_obses'].low.shape
    robot_state_shape = source_env.observation_space['robot_states'].low.shape
    assert robot_state_shape == target_env.observation_space['robot_states'].low.shape
    action_shape = source_env.action_space.low.shape
    assert action_shape == target_env.action_space.low.shape

    use_state_input = variant["use_state_input"]
    vae_latent_dim = variant['vae_model_kwargs']['representation_size']
    action_dim = action_shape[0]
    q_input_dim = vae_latent_dim + action_dim
    policy_input_dim = vae_latent_dim
    if use_state_input:
        q_input_dim += robot_state_shape[0]
        policy_input_dim += robot_state_shape[0]

    src_qf1 = ConcatMlp(input_size=q_input_dim, output_size=1, **variant['qf_kwargs'])
    src_qf2 = ConcatMlp(input_size=q_input_dim, output_size=1, **variant['qf_kwargs'])
    src_target_qf1 = ConcatMlp(input_size=q_input_dim, output_size=1, **variant['qf_kwargs'])
    src_target_qf2 = ConcatMlp(input_size=q_input_dim, output_size=1, **variant['qf_kwargs'])

    trg_qf1 = ConcatMlp(input_size=q_input_dim, output_size=1, **variant['qf_kwargs'])
    trg_qf2 = ConcatMlp(input_size=q_input_dim, output_size=1, **variant['qf_kwargs'])
    trg_target_qf1 = ConcatMlp(input_size=q_input_dim, output_size=1, **variant['qf_kwargs'])
    trg_target_qf2 = ConcatMlp(input_size=q_input_dim, output_size=1, **variant['qf_kwargs'])

    # Instantiate trainer with appropriate agent
    src_agent = TanhGaussianPolicy(
            obs_dim=policy_input_dim,
            action_dim=action_dim,
            **variant['policy_kwargs']
    )
    trg_agent = TanhGaussianPolicy(
            obs_dim=policy_input_dim,
            action_dim=action_dim,
            **variant['policy_kwargs']
    )
    src_policy_trainer = CQLTrainer(
            action_shape=(action_dim,),
            policy=src_agent,
            qf1=src_qf1,
            qf2=src_qf2,
            target_qf1=src_target_qf1,
            target_qf2=src_target_qf2,
            use_state_input=use_state_input,
            **variant['policy_trainer_kwargs']
    )
    trg_policy_trainer = CQLTrainer(
            action_shape=(action_dim,),
            policy=trg_agent,
            qf1=trg_qf1,
            qf2=trg_qf2,
            target_qf1=trg_target_qf1,
            target_qf2=trg_target_qf2,
            use_state_input=use_state_input,
            **variant['policy_trainer_kwargs']
    )

    # Transfer Module and Trainer
    vae_model = InvVAE(
        image_shape=image_obs_shape,
        action_shape=action_shape,
        **variant['vae_model_kwargs']
    )

    # Eval Agent, Pay Attention to the Agent Assignment
    eval_source_agent = VAEMakeDeterministic(src_agent, vae_model, use_state_input, 'source')
    eval_target_agent = VAEMakeDeterministic(trg_agent, vae_model, use_state_input, 'target')

    # Datasets and Buffers
    src_train_buffer = RobotSharingReplayDataset(
        robot_state_shape[0], vae_latent_dim, action_shape[0],
        variant['algorithm_kwargs']['policy_buffer_size']
    )
    trg_train_buffer = RobotSharingReplayDataset(
        robot_state_shape[0], vae_latent_dim, action_shape[0],
        variant['algorithm_kwargs']['policy_buffer_size']
    )
    eval_source_collector = MdpPathCollector(source_env, eval_source_agent, rollout_fn=robo_rollout)
    eval_target_collector = MdpPathCollector(target_env, eval_target_agent, rollout_fn=robo_rollout)
    src_general_dataset = TrjRoboDataset(**variant['source_general_dataset_kwargs'])
    trg_general_dataset = TrjRoboDataset(**variant['target_general_dataset_kwargs'])
    src_supervise_dataset = TrjRoboDataset(**variant['source_supervise_dataset_kwargs'])
    trg_supervise_dataset = TrjRoboDataset(**variant['target_supervise_dataset_kwargs'])
    src_train_dataset = TrjRoboDataset(**variant['source_train_dataset_kwargs'])
    trg_train_dataset = TrjRoboDataset(**variant['target_train_dataset_kwargs'])

    # Define algorithm
    algorithm = OfflineInvVAESharingAlgorithm(
        source_agent=src_agent,
        target_agent=trg_agent,
        transfer_model=vae_model,

        source_trainer=src_policy_trainer,
        target_trainer=trg_policy_trainer,

        eval_source_env=source_env,
        eval_target_env=target_env,
        eval_source_collector=eval_source_collector,
        eval_target_collector=eval_target_collector,

        source_general_dataset=src_general_dataset,
        target_general_dataset=trg_general_dataset,
        source_supervise_dataset=src_supervise_dataset,
        target_supervise_dataset=trg_supervise_dataset,
        source_train_dataset=src_train_dataset,
        target_train_dataset=trg_train_dataset,
        source_train_buffer=src_train_buffer,
        target_train_buffer=trg_train_buffer,

        save_model_path=variant['checkpoint_dir'],
        **variant['algorithm_kwargs']
    )

    algorithm.to(ptu.device)
    algorithm.load_transfer_model(variant['algorithm_kwargs']["transfer_model_path"])
    if variant['load_checkpoint']:
        algorithm.load_snapshot()
    algorithm.train()


def deep_update_dict(fr, to):
    ''' update dict of dicts with new variants '''
    for k, v in fr.items():
        if type(v) is dict and k in to:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to


