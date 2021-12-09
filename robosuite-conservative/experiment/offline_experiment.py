
import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import ActionNormalizedEnv
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.samplers.rollout_functions import robo_rollout
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.data_management.custom_replay_buffer import CustomRobotReplayBuffer
from models.combo.combo_trainer import ComboTrainer
from models.combo.agent import ComboAgent
from models.dataset import TrjRoboDataset
from models.img_cql.cql_trainer import IMGCQLTrainer
from models.cql.cql import CQLTrainer, StMakeDeterministic
from models.img_cql.networks import Actor, Critic, IMGMakeDeterministic


from experiment.offline_algorithm import OfflineBatchIMGRLAlgorithm, \
    OfflineBatchLatentRLAlgorithm, OfflineBatchRLAlgorithm

import robosuite as suite
from robosuite.wrappers import GymVisualWrapper, GymVisualCatWrapper
from robosuite.controllers import load_controller_config, ALL_CONTROLLERS


def offline_latent_experiment(variant):

    env_config = variant["environment_kwargs"]
    wrapper_config = variant["wrapper_kwargs"]

    controller = env_config.pop("controller")
    if controller in set(ALL_CONTROLLERS):
        controller_config = load_controller_config(default_controller=controller)
    else:
        controller_config = load_controller_config(custom_fpath=controller)

    # Create envs
    env = suite.make(**env_config, has_renderer=False, controller_configs=controller_config)
    env = ActionNormalizedEnv(GymVisualWrapper(env, **wrapper_config))

    # (H, W, C)
    image_obs_shape = env.observation_space['image_obses'].low.shape
    image_obs_shape = image_obs_shape[-1:] + image_obs_shape[:-1]
    robot_state_shape = env.observation_space['robot_states'].low.shape
    action_shape = env.action_space.low.shape

    agent = ComboAgent(
        image_shape=image_obs_shape,
        state_shape=robot_state_shape,
        action_shape=action_shape,
        reward_shape=(1,),
        value_shape=(1,),
        device=ptu.device,
        **variant['agent_kwargs']
    )

    trainer = ComboTrainer(
        agent=agent,
        device=ptu.device,
        **variant['trainer_kwargs']
    )

    eval_path_collector = MdpPathCollector(env, agent, rollout_fn=robo_rollout)
    dataset = TrjRoboDataset(**variant['dataset_kwargs'])

    # Define algorithm
    algorithm = OfflineBatchLatentRLAlgorithm(
        device=ptu.device,
        dataset=dataset,
        agent=agent,
        trainer=trainer,
        evaluation_env=env,
        evaluation_data_collector=eval_path_collector,
        **variant['algorithm_kwargs']
    )

    if variant['load_model_snapshot']:
        algorithm.load_model(variant['load_snapshot_path'])
    algorithm.train()


def offline_img_cql_experiment(variant):

    env_config = variant["environment_kwargs"]
    wrapper_config = variant["wrapper_kwargs"]

    controller = env_config.pop("controller")
    if controller in set(ALL_CONTROLLERS):
        controller_config = load_controller_config(default_controller=controller)
    else:
        controller_config = load_controller_config(custom_fpath=controller)

    # Create envs
    env = suite.make(**env_config, has_renderer=False, controller_configs=controller_config)
    env = ActionNormalizedEnv(GymVisualWrapper(env, **wrapper_config))

    # (H, W, C)
    image_obs_shape = env.observation_space['image_obses'].low.shape
    robot_state_shape = env.observation_space['robot_states'].low.shape
    action_shape = env.action_space.low.shape

    qf = Critic(
        image_obs_shape=image_obs_shape,
        state_shape=robot_state_shape,
        action_shape=action_shape,
        **variant['qf_kwargs'],
        **variant['encoder_kwargs']
    )
    target_qf = Critic(
        image_obs_shape=image_obs_shape,
        state_shape=robot_state_shape,
        action_shape=action_shape,
        **variant['qf_kwargs'],
        **variant['encoder_kwargs']
    )

    # Instantiate trainer with appropriate agent
    agent = Actor(
        image_obs_shape=image_obs_shape,
        state_shape=robot_state_shape,
        action_shape=action_shape,
        **variant['policy_kwargs'],
        **variant['encoder_kwargs']
    )
    eval_agent = IMGMakeDeterministic(agent)
    trainer = IMGCQLTrainer(
        env=env,
        policy=agent,
        qf=qf,
        target_qf=target_qf,
        device=ptu.device,
        **variant['trainer_kwargs']
    )

    eval_path_collector = MdpPathCollector(env, eval_agent, rollout_fn=robo_rollout)
    dataset = TrjRoboDataset(**variant['dataset_kwargs'])

    # Define algorithm
    algorithm = OfflineBatchIMGRLAlgorithm(
        device=ptu.device,
        dataset=dataset,
        agent=agent,
        trainer=trainer,
        evaluation_env=env,
        evaluation_data_collector=eval_path_collector,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


def offline_cql_experiment(variant):

    env_config = variant["environment_kwargs"]
    wrapper_config = variant["wrapper_kwargs"]

    controller = env_config.pop("controller")
    if controller in set(ALL_CONTROLLERS):
        controller_config = load_controller_config(default_controller=controller)
    else:
        controller_config = load_controller_config(custom_fpath=controller)

    # Create envs
    env = suite.make(**env_config, has_renderer=False, controller_configs=controller_config)
    # Todo Here use VisualCatWrapper
    env = ActionNormalizedEnv(GymVisualCatWrapper(env, **wrapper_config))

    obs_dim = env.observation_space['robot_states'].low.size + \
        env.observation_space['object_states'].low.size
    action_dim = env.action_space.low.size

    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )

    # Instantiate trainer with appropriate agent
    agent = TanhGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            **variant['policy_kwargs'],
    )
    eval_policy = StMakeDeterministic(agent)
    trainer = CQLTrainer(
            env=env,
            policy=agent,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            **variant['trainer_kwargs']
    )

    replay_buffer = CustomRobotReplayBuffer(
        env,
        variant['trainer_kwargs']['buffer_size'],
    )
    eval_path_collector = MdpPathCollector(
        env,
        eval_policy,
        rollout_fn=robo_rollout
    )
    dataset = TrjRoboDataset(**variant['dataset_kwargs'])

    # Define algorithm
    algorithm = OfflineBatchRLAlgorithm(
        device=ptu.device,
        dataset=dataset,
        agent=agent,
        trainer=trainer,
        evaluation_env=env,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


def deep_update_dict(fr, to):
    ''' update dict of dicts with new variants '''
    for k, v in fr.items():
        if type(v) is dict and k in to:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to


