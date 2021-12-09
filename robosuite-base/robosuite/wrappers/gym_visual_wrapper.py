"""
This file implements a wrapper for facilitating compatibility with OpenAI gym.
This is useful when using these environments with code that assumes a gym-like 
interface.
"""

import numpy as np
from gym import spaces
from gym.core import Env
from robosuite.wrappers import Wrapper
from collections import OrderedDict


class GymVisualWrapper(Wrapper, Env):
    """
    Initializes the Gym Visual wrapper. Mimics many of the required functionalities of the Wrapper class
    found in the gym.core module

    Args:
        env (MujocoEnv): The environment to wrap.
        state_keys (None or list of str): If provided, each robot state will
            consist of items from the wrapped environment's
            observation dictionary. Defaults to proprio-state.
        visual_keys (None or str): If provided, each visual observation returns the image
            of the visual_keys. Defaults to frontview image of the robots.
        object_keys (None or list of str): If provided, each object state will
            consist of object states from the wrapped environment's
            observation dictionary. Defaults to object-state.


    Raises:
        AssertionError: [Object observations must be enabled if object_keys]
    """

    def __init__(self, env, state_keys=None, visual_keys=None, object_keys=None):
        # Run super method
        super().__init__(env=env)
        # Create name for gym
        robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
        self.name = robots + "_" + type(self.env).__name__

        # Get reward range
        self.reward_range = (0, self.env.reward_scale)

        if state_keys is None:
            state_keys = []
            # Iterate over all robots to add to state
            for idx in range(len(self.env.robots)):
                state_keys += ["robot{}_proprio-state".format(idx)]

        if visual_keys is None:
            visual_keys = "frontview_image"

        if object_keys is None:
            object_keys = []
            if self.env.use_object_obs:
                object_keys += ["object-state"]

        self.state_keys = state_keys
        self.visual_keys = visual_keys
        self.object_keys = object_keys

        # Gym specific attributes
        self.env.spec = None
        self.metadata = None

        # set up observation and action spaces
        obs = self.env.reset()
        self.modality_dims = {key: obs[key].shape for key in
                              (self.state_keys + self.object_keys + [self.visual_keys])}
        obs = self._process_obs(obs)
        self.obs_dim = OrderedDict()
        self.observation_space = OrderedDict()
        for k, v in obs.items():
            self.obs_dim[k] = v.size
            high = np.inf * np.ones(v.shape)
            low = -high
            self.observation_space[k] = spaces.Box(low=low, high=high)
        low, high = self.env.action_spec
        self.action_space = spaces.Box(low=low, high=high)

    def _process_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed

        Returns:
            np.array: processed observation dictionary, includes: robot_state,
            object_state (should not use), image_obs
        """
        state_lst = []
        for key in self.state_keys:
            if key in obs_dict:
                if verbose:
                    print("adding key: {}".format(key))
                state_lst.append(np.array(obs_dict[key]).flatten())

        object_lst = []
        for key in self.object_keys:
            if key in obs_dict:
                if verbose:
                    print("adding key: {}".format(key))
                object_lst.append(np.array(obs_dict[key]).flatten())

        state_lst = np.concatenate(state_lst)
        object_lst = np.concatenate(object_lst)
        visual_lst = obs_dict[self.visual_keys]

        obs = OrderedDict()
        obs["robot_state"] = state_lst
        obs["object_state"] = object_lst
        obs["image_obs"] = visual_lst

        return obs

    def reset(self):
        """
        Extends env reset method to return processed observation instead of normal OrderedDict.

        Returns:
            np.array: Processed environment observation space after reset occurs
        """
        ob_dict = self.env.reset()
        return self._process_obs(ob_dict)

    def step(self, action):
        """
        Extends vanilla step() function call to return processed observation instead of normal OrderedDict.

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:
                - (dict) observations from the environment, include robot_state,
                    object_state, image_obs
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        ob_dict, reward, done, info = self.env.step(action)
        return self._process_obs(ob_dict), reward, done, info

    def seed(self, seed=None):
        """
        Utility function to set numpy seed

        Args:
            seed (None or int): If specified, numpy seed to set

        Raises:
            TypeError: [Seed must be integer]
        """
        # Seed the generator
        if seed is not None:
            try:
                np.random.seed(seed)
            except:
                TypeError("Seed must be an integer type!")

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Dummy function to be compatible with gym interface that simply returns environment reward

        Args:
            achieved_goal: [NOT USED]
            desired_goal: [NOT USED]
            info: [NOT USED]

        Returns:
            float: environment reward
        """
        # Dummy args used to mimic Wrapper interface
        return self.env.reward()
