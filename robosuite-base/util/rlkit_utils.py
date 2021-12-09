
from rlkit.torch.pytorch_util import set_gpu_mode

import torch
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.img_replay_buffer import VisualRobotReplayBuffer
from rlkit.envs.wrappers import ActionNormalizedEnv
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac_img.sac_img import SACIMGTrainer
from rlkit.torch.sac_img.networks import Actor, Critic, MakeDeterministic
from util.rlkit_custom import CustomTorchBatchRLAlgorithm, rollout

from rlkit.core import logger
import robosuite as suite
from robosuite.wrappers import GymVisualWrapper

from robosuite.controllers import load_controller_config, ALL_CONTROLLERS

import numpy as np


# Define agents available
AGENTS = {"SACIMG"}


def experiment(variant, agent="SACIMG"):

    # Make sure agent is a valid choice
    assert agent in AGENTS, "Invalid agent selected. Selected: {}. Valid options: {}".format(agent, AGENTS)

    env_config = variant["environment_kwargs"]

    controller = env_config.pop("controller")
    if controller in set(ALL_CONTROLLERS):
        controller_config = load_controller_config(default_controller=controller)
    else:
        controller_config = load_controller_config(custom_fpath=controller)

    img_size = variant["image_size"]
    env_config["camera_heights"] = img_size
    env_config["camera_widths"] = img_size

    # Create envs
    expl_env = suite.make(**env_config, has_renderer=False, controller_configs=controller_config)
    expl_env = ActionNormalizedEnv(GymVisualWrapper(expl_env, visual_keys=env_config["camera_names"]+"_image"))
    eval_env = suite.make(**env_config, has_renderer=False, controller_configs=controller_config)
    eval_env = ActionNormalizedEnv(GymVisualWrapper(eval_env, visual_keys=env_config["camera_names"] + "_image"))

    # (H, W, C)
    image_obs_dim = expl_env.observation_space['image_obs'].low.shape
    robot_state_dim = expl_env.observation_space['robot_state'].low.size
    action_dim = expl_env.action_space.low.size

    qf = Critic(
        image_obs_shape=image_obs_dim,
        state_shape=robot_state_dim,
        action_shape=action_dim,
        **variant['qf_kwargs'],
        **variant['encoder_kwargs']
    )
    target_qf = Critic(
        image_obs_shape=image_obs_dim,
        state_shape=robot_state_dim,
        action_shape=action_dim,
        **variant['qf_kwargs'],
        **variant['encoder_kwargs']
    )

    # Define references to variables that are agent-specific
    trainer = None
    eval_policy = None
    expl_policy = None

    # Instantiate trainer with appropriate agent
    if agent == "SACIMG":
        expl_policy = Actor(
            image_obs_shape=image_obs_dim,
            state_shape=robot_state_dim,
            action_shape=action_dim,
            **variant['policy_kwargs'],
            **variant['encoder_kwargs']
        )
        eval_policy = MakeDeterministic(expl_policy)
        trainer = SACIMGTrainer(
            env=eval_env,
            policy=expl_policy,
            qf=qf,
            target_qf=target_qf,
            **variant['trainer_kwargs']
        )
    else:
        print("Error: No valid agent chosen!")

    replay_buffer = VisualRobotReplayBuffer(
        expl_env,
        variant['replay_buffer_size'],
    )
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        expl_policy,
    )

    # Define algorithm
    algorithm = CustomTorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


def evaluate_policy(env_config, model_path, n_eval, printout=False):
    if printout:
        print("Loading policy...")

    # Load trained model and corresponding policy
    data = torch.load(model_path)
    policy = data['evaluation/policy']

    if printout:
        print("Policy loaded")

    # Load controller
    controller = env_config.pop("controller")
    if controller in set(ALL_CONTROLLERS):
        # This is a default controller
        controller_config = load_controller_config(default_controller=controller)
    else:
        # This is a string to the custom controller
        controller_config = load_controller_config(custom_fpath=controller)

    # Create robosuite env
    env = suite.make(**env_config,
                     has_renderer=False,
                     has_offscreen_renderer=False,
                     use_object_obs=True,
                     use_camera_obs=False,
                     reward_shaping=True,
                     controller_configs=controller_config
                     )
    env = GymWrapper(env)

    # Use CUDA if available
    if torch.cuda.is_available():
        set_gpu_mode(True)
        policy.cuda() if not isinstance(policy, MakeDeterministic) else policy.stochastic_policy.cuda()

    if printout:
        print("Evaluating policy over {} simulations...".format(n_eval))

    # Create variable to hold rewards to be averaged
    returns = []

    # Loop through simulation n_eval times and take average returns each time
    for i in range(n_eval):
        path = rollout(
            env,
            policy,
            max_path_length=env_config["horizon"],
            render=False,
        )

        # Determine total summed rewards from episode and append to 'returns'
        returns.append(sum(path["rewards"]))

    # Average the episode rewards and return the normalized corresponding value
    return sum(returns) / (env_config["reward_scale"] * n_eval)

