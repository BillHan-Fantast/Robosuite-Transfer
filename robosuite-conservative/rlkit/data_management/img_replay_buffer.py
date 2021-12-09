from gym.spaces import Discrete

from rlkit.envs.env_utils import get_dim
import numpy as np
from collections import OrderedDict
import warnings

from rlkit.data_management.replay_buffer import ReplayBuffer

class VisualRobotReplayBuffer(ReplayBuffer):

    def __init__(
        self,
        env,
        max_replay_buffer_size,
        env_info_sizes=None,
        replace=True,
    ):
        self.observation_space, self.action_space = env.observation_space, env.action_space
        self.image_obs_shape = (-1,) + self.observation_space['image_obs'].shape
        self._image_obs_dim = get_dim(self.observation_space['image_obs'])
        self._robot_state_dim = get_dim(self.observation_space['robot_state'])
        self._object_state_dim = get_dim(self.observation_space['object_state'])
        self._action_dim = get_dim(self.action_space)
        self._max_replay_buffer_size = max_replay_buffer_size
        self._image_obs = np.zeros((max_replay_buffer_size, self._image_obs_dim), dtype='uint8')
        self._robot_state = np.zeros((max_replay_buffer_size, self._robot_state_dim))
        self._object_state = np.zeros((max_replay_buffer_size, self._object_state_dim))
        self._next_image_obs = np.zeros((max_replay_buffer_size, self._image_obs_dim), dtype='uint8')
        self._next_robot_state = np.zeros((max_replay_buffer_size, self._robot_state_dim))
        self._next_object_state = np.zeros((max_replay_buffer_size, self._object_state_dim))
        self._actions = np.zeros((max_replay_buffer_size, self._action_dim))
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        self._env_infos = {}
        for key, size in env_info_sizes.items():
            self._env_infos[key] = np.zeros((max_replay_buffer_size, size))
        self._env_info_keys = list(env_info_sizes.keys())

        self._replace = replace

        self._top = 0
        self._size = 0

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, env_info, **kwargs):
        if isinstance(self.action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action

        self._image_obs[self._top] = observation['image_obs'].flatten().copy()
        self._robot_state[self._top] = observation['robot_state'].copy()
        self._object_state[self._top] = observation['object_state'].copy()
        self._actions[self._top] = new_action.copy()
        self._rewards[self._top] = reward.copy()
        self._terminals[self._top] = terminal.copy()
        self._next_image_obs[self._top] = next_observation['image_obs'].flatten().copy()
        self._next_robot_state[self._top] = next_observation['robot_state'].copy()
        self._next_object_state[self._top] = next_observation['object_state'].copy()

        for key in self._env_info_keys:
            self._env_infos[key][self._top] = env_info[key].copy()
        self._advance()

    def terminate_episode(self):
        pass

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def random_batch(self, batch_size):
        indices = np.random.choice(self._size, size=batch_size, replace=self._replace or self._size < batch_size)
        if not self._replace and self._size < batch_size:
            warnings.warn('Replace was set to false, but is temporarily set to true '
                          'because batch size is larger than current size of replay.')
        batch = dict(
            image_obses=self._image_obs[indices].reshape(self.image_obs_shape),
            robot_states=self._robot_state[indices],
            object_states=self._object_state[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_image_obses=self._next_image_obs[indices].reshape(self.image_obs_shape),
            next_robot_states=self._next_robot_state[indices],
            next_object_states=self._next_object_state[indices]
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        return batch

    def rebuild_env_info_dict(self, idx):
        return {
            key: self._env_infos[key][idx]
            for key in self._env_info_keys
        }

    def batch_env_info_dict(self, indices):
        return {
            key: self._env_infos[key][indices]
            for key in self._env_info_keys
        }

    def num_steps_can_sample(self):
        return self._size

    def get_diagnostics(self):
        return OrderedDict([
            ('size', self._size)
        ])

