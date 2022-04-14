
import h5py
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader


class TrjRoboDataset(Dataset):

    def __init__(self, dataset_path, episode_skip=1, model_batch_length=50, max_epi_length=1000, **kwargs):

        if os.path.exists('/scr/beining/' + dataset_path):
            prefix = '/scr/beining'
        elif os.path.exists('/scr-ssd/beining/' + dataset_path):
            prefix = '/scr-ssd/beining'
        else:
            prefix = '../'

        self.dataset_path = os.path.join(prefix, dataset_path)
        self.model_batch_length = model_batch_length
        self.episode_skip = episode_skip
        file = h5py.File(self.dataset_path, 'r')
        all_episodes = list(file.keys())
        self.episodes = all_episodes[::episode_skip]
        self.max_epi_length = max_epi_length
        self.num_epis = len(self.episodes)
        file.close()

    def __len__(self):
        return self.num_epis

    def __getitem__(self, index):
        file = h5py.File(self.dataset_path, 'r')
        episode = file[self.episodes[index]]

        total_steps = min(len(episode['terminals'][()]), self.max_epi_length)
        available_steps = int(total_steps) - self.model_batch_length
        assert available_steps >= 0, "Only support full length episodes"
        index = int(np.random.randint(0, available_steps))

        episode = {k: v[index: index + self.model_batch_length] for k, v in episode.items()}

        file.close()
        return episode

    def get_episode(self, index):
        """
        Return full episodes
        """
        file = h5py.File(self.dataset_path, 'r')
        episode = file[self.episodes[index]]
        total_steps = min(len(episode['terminals'][()]), self.max_epi_length)
        episode = {k: v[:total_steps] for k, v in episode.items()}
        file.close()
        return episode


class StepRoboDataset(Dataset):

    def __init__(self, dataset_path, episode_skip=1, max_epi_length=1000, **kwargs):
        if os.path.exists('/scr/beining/' + dataset_path):
            prefix = '/scr/beining'
        elif os.path.exists('/scr-ssd/beining/' + dataset_path):
            prefix = '/scr-ssd/beining'
        else:
            prefix = '../'

        self.dataset_path = os.path.join(prefix, dataset_path)
        self.episode_skip = episode_skip
        file = h5py.File(self.dataset_path, 'r')
        all_episodes = list(file.keys())
        self.episodes = all_episodes[::episode_skip]
        self.max_epi_length = min(max_epi_length, len(file[self.episodes[0]]['robot_states'])) - 1
        self.num_epis = len(self.episodes)
        self.num_steps = self.num_epis * self.max_epi_length
        file.close()

    def __len__(self):
        return self.num_steps

    def __getitem__(self, index):
        file = h5py.File(self.dataset_path, 'r')
        epi, step = index // self.max_epi_length, index % self.max_epi_length
        episode = file[self.episodes[epi]]
        episode = {k: v[step: step+2] for k, v in episode.items()}
        file.close()
        return episode


class CustomDataLoader(object):
    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False, num_workers=0, collate_fn=None):
        self.loader = DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=drop_last,
                                 num_workers=num_workers, collate_fn=collate_fn)

        self._length = len(self.loader)
        self._step = 0
        self._iterator = iter(self.loader)

    def _reset_loader(self):
        self._step = 0
        self._iterator = iter(self.loader)

    def sample(self):
        if self._step >= self._length:
            self._reset_loader()

        batch = next(self._iterator)
        self._step += 1

        return batch