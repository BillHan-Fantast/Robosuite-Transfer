
import h5py
import numpy as np
from torch.utils.data import Dataset


class TrjRoboDataset(Dataset):

    def __init__(self, dataset_path, episode_skip=1, model_batch_length=50, max_epi_length=1000, **kwargs):
        self.dataset_path = dataset_path
        self.model_batch_length = model_batch_length
        self.episode_skip = episode_skip
        file = h5py.File(dataset_path, 'r')
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
        self.dataset_path = dataset_path
        self.episode_skip = episode_skip
        file = h5py.File(dataset_path, 'r')
        all_episodes = list(file.keys())
        self.episodes = all_episodes[::episode_skip]
        self.max_epi_length = min(max_epi_length, len(file[self.episodes[0]]['terminals'])) - 1
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

# Test robosuite dataset object

if __name__ == '__main__':

    print('------------- Testing --------------')

    path = '/home/hbn/Desktop/Robot-Transfer/dataset/robosuite/Lift-Panda-OSC-POSE-default/medium_expert.hdf5'

    dataset = TrjRoboDataset(path, episode_skip=2, model_batch_length=50)

    print(len(dataset))
    epi = dataset[10]

    for k, v in epi.items():
        print(k+':')
        print(v)
        print('Length: '+str(len(v)))

    print('Test Success!')