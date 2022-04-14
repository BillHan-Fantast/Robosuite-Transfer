
import numpy as np
from collections import OrderedDict
import warnings


class RobotReplayDataset(object):

    def __init__(
        self,
        robot_state_dim,
        object_state_dim,
        action_dim,
        max_replay_buffer_size,
        replace=True,
    ):
        self._robot_state_dim = robot_state_dim
        self._object_state_dim = object_state_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self._robot_state = np.zeros((max_replay_buffer_size, self._robot_state_dim))
        self._object_state = np.zeros((max_replay_buffer_size, self._object_state_dim))
        self._next_robot_state = np.zeros((max_replay_buffer_size, self._robot_state_dim))
        self._next_object_state = np.zeros((max_replay_buffer_size, self._object_state_dim))
        self._actions = np.zeros((max_replay_buffer_size, self._action_dim))
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')

        self._replace = replace

        self._top = 0
        self._size = 0

    def add_sample(self, observation, action, reward, terminal, next_observation, **kwargs):

        self._robot_state[self._top] = observation['robot_states'].copy()
        self._object_state[self._top] = observation['object_states'].copy()
        self._actions[self._top] = action.copy()
        self._rewards[self._top] = reward.copy()
        self._terminals[self._top] = terminal.copy()
        self._next_robot_state[self._top] = next_observation['robot_states'].copy()
        self._next_object_state[self._top] = next_observation['object_states'].copy()

        self._advance()

    def add_path(self, path):
        for i, (
                obs,
                action,
                reward,
                next_obs,
                terminal,
        ) in enumerate(zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["terminals"],
        )):
            self.add_sample(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                terminal=terminal,
            )
        self.terminate_episode()

    def add_paths(self, paths):
        for path in paths:
            self.add_path(path)

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
            robot_states=self._robot_state[indices],
            object_states=self._object_state[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_robot_states=self._next_robot_state[indices],
            next_object_states=self._next_object_state[indices]
        )
        return batch

    def num_steps_can_sample(self):
        return self._size

    def get_diagnostics(self):
        return OrderedDict([
            ('size', self._size)
        ])


class RobotSharingReplayDataset(object):

    def __init__(
        self,
        robot_state_dim,
        object_state_dim,
        action_dim,
        buffer_size,
        replace=True,
    ):
        self.transfer_replay_buffer = RobotReplayDataset(robot_state_dim, object_state_dim,
                                                         action_dim, buffer_size, replace)
        self.origin_sup_replay_buffer = RobotReplayDataset(robot_state_dim, object_state_dim,
                                                           action_dim, buffer_size, replace)
        self.origin_unsup_replay_buffer = RobotReplayDataset(robot_state_dim, object_state_dim,
                                                             action_dim, buffer_size, replace)

    def add_path(self, path, type):

        if type == 'transfer':
            self.transfer_replay_buffer.add_path(path)
        elif type == 'origin_sup':
            self.origin_sup_replay_buffer.add_path(path)
        elif type == 'origin_unsup':
            self.origin_unsup_replay_buffer.add_path(path)
        else:
            raise NotImplementedError

    def random_batch(self, transfer_batch_size, origin_sup_batch_size, origin_unsup_batch_size):
        transfer_batch, origin_sup_batch, origin_unsup_batch = None, None, None
        if transfer_batch_size > 0:
            transfer_batch = self.transfer_replay_buffer.random_batch(transfer_batch_size)
        if origin_sup_batch_size > 0:
            origin_sup_batch = self.origin_sup_replay_buffer.random_batch(origin_sup_batch_size)
        if origin_unsup_batch_size > 0:
            origin_unsup_batch = self.origin_unsup_replay_buffer.random_batch(origin_unsup_batch_size)

        batch = merge_batch(transfer_batch, origin_sup_batch)
        batch = merge_batch(batch, origin_unsup_batch)

        return batch


def merge_batch(src, trg):
    if src is None:
        return trg
    elif trg is None:
        return src
    else:
        batch = dict()
        for k, v in src.items():
            batch[k] = np.concatenate([v, trg[k]], axis=0)
        return batch
