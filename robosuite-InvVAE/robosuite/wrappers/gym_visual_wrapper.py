"""
This file implements a wrapper for facilitating compatibility with OpenAI gym.
This is useful when using these environments with code that assumes a gym-like 
interface.
"""

import numpy as np
import cv2
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
        image_process_kwargs: The functional parameters to crop the visual inputs.

    Raises:
        AssertionError: [Object observations must be enabled if object_keys]
    """

    def __init__(self, env, state_keys=None, visual_keys=None, object_keys=None, image_process_kwargs={}):
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
        self.proprio_keys = ["robot{}_proprio-state".format(idx)
                             for idx in range(len(self.env.robots))]

        # Gym specific attributes
        self.env.spec = None
        self.metadata = None

        # Image process parameters
        self.image_crop_yx = image_process_kwargs['image_crop_yx']
        self.image_crop_size = image_process_kwargs['image_crop_size']
        self.image_obs_size = image_process_kwargs['image_obs_size']  # 84
        assert self.image_crop_size >= self.image_obs_size

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
            np.array: processed observation dictionary, includes: robot_states,
            object_states, image_obses, proprio_states, raw_images
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

        proprio_lst = []
        for key in self.proprio_keys:
            if key in obs_dict:
                if verbose:
                    print("adding key: {}".format(key))
                proprio_lst.append(np.array(obs_dict[key]).flatten())

        state_lst = np.concatenate(state_lst)
        object_lst = np.concatenate(object_lst)
        proprio_lst = np.concatenate(proprio_lst)
        image_lst = obs_dict[self.visual_keys]

        if image_lst.shape[0] == self.image_obs_size:
            visual_lst = image_lst
        else:
            visual_lst = self._process_image(image_lst)

        obs = OrderedDict()
        obs["robot_states"] = state_lst
        obs["object_states"] = object_lst
        obs["image_obses"] = visual_lst
        obs["proprio_states"] = proprio_lst
        for k, v in obs_dict.items():
            if k.endswith('image'):
                obs[k] = v

        return obs

    def _process_image(self, img):
        """
        Process the images to the desired part and shape.
        """
        assert (img.shape[0] >= self.image_crop_size) and (img.shape[1] >= self.image_crop_size)

        y, x = self.image_crop_yx["y"], self.image_crop_yx["x"]
        img = img[y:y+self.image_crop_size, x:x+self.image_crop_size]
        img = cv2.resize(img, (self.image_obs_size, self.image_obs_size), interpolation=cv2.INTER_AREA)

        return img

    def process_images(self, imgs):
        """
        Process the images, called by functions outside.
        """
        assert len(imgs.shape) == 4 \
               and (imgs.shape[1] >= self.image_crop_size) \
               and (imgs.shape[2] >= self.image_crop_size)

        post_imgs = []
        for img in imgs:
            post_imgs.append(self._process_image(img))

        return np.array(post_imgs)

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

