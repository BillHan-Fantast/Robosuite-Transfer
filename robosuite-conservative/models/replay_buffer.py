import h5py
import numpy as np


class ReplayBuffer(object):
    def __init__(self, size: int, obs_dim: int, action_dim: int, immutable: bool = False):
        self.immutable = immutable
        self.max_size = size

        self._obs = np.full((size, obs_dim), float('nan'), dtype=np.float32)
        self._actions = np.full((size, action_dim), float('nan'), dtype=np.float32)
        self._rewards = np.full((size, 1), float('nan'), dtype=np.float32)
        self._next_obs = np.full((size, obs_dim), float('nan'), dtype=np.float32)
        self._terminals = np.full((size, 1), float('nan'), dtype=np.float32)

        self._stored_steps = 0
        self._write_location = 0

    @property
    def obs_dim(self):
        return self._obs.shape[-1]

    @property
    def action_dim(self):
        return self._actions.shape[-1]

    def __len__(self):
        return self._stored_steps

    def save(self, location: str):
        f = h5py.File(location+'.hdf5', 'w')
        f.create_dataset('obs', data=self._obs[:self._stored_steps], compression='lzf')
        f.create_dataset('actions', data=self._actions[:self._stored_steps], compression='lzf')
        f.create_dataset('rewards', data=self._rewards[:self._stored_steps], compression='lzf')
        f.create_dataset('next_obs', data=self._next_obs[:self._stored_steps], compression='lzf')
        f.create_dataset('terminals', data=self._terminals[:self._stored_steps], compression='lzf')
        f.close()

    def load(self, location: str):
        with h5py.File(location+'.hdf5', "r") as f:
            obs = np.array(f['obs'])
            self._stored_steps = obs.shape[0]
            self._write_location = obs.shape[0] % self.max_size
            self._obs[:self._stored_steps] = np.array(f['obs'])
            self._actions[:self._stored_steps] = np.array(f['actions'])
            self._rewards[:self._stored_steps] = np.array(f['rewards'])
            self._next_obs[:self._stored_steps] = np.array(f['next_obs'])
            self._terminals[:self._stored_steps] = np.array(f['terminals'])

    def add_samples(self, obs_feats, actions, next_obs_feats, rewards, terminals):
        for obsi, actsi, nobsi, rewi, termi in zip(obs_feats, actions, next_obs_feats, rewards, terminals):
            self._obs[self._write_location] = obsi
            self._actions[self._write_location] = actsi
            self._next_obs[self._write_location] = nobsi
            self._rewards[self._write_location] = rewi
            self._terminals[self._write_location] = termi

            self._write_location = (self._write_location + 1) % self.max_size
            self._stored_steps = min(self._stored_steps + 1, self.max_size)

    def sample(self, batch_size, return_dict: bool = False):
        idxs = np.random.randint(self._stored_steps, size=batch_size)

        data = {
            'obs': self._obs[idxs].copy(),
            'actions': self._actions[idxs].copy(),
            'next_obs': self._next_obs[idxs].copy(),
            'rewards': self._rewards[idxs].copy(),
            'terminals': self._terminals[idxs].copy()
        }

        return data


class LatentReplayBuffer(object):
    def __init__(self, 
                 real_size: int, 
                 latent_size: int, 
                 obs_dim: int, 
                 action_dim: int,
                 immutable: bool = False
                 ):
        
        self.immutable = immutable
        self.real_buffer = ReplayBuffer(real_size, obs_dim, action_dim, immutable)
        self.latent_buffer = ReplayBuffer(latent_size, obs_dim, action_dim, immutable)

        self.max_size = real_size + latent_size
        
    @property
    def obs_dim(self):
        return self.real_buffer.obs_dim

    @property
    def action_dim(self):
        return self.real_buffer.action_dim

    def __len__(self):
        return self.real_buffer._stored_steps + self.latent_buffer._stored_steps

    def save(self, location: str):
        # only store real samples
        self.real_buffer.save(location+'_real')
        
    def load(self, location: str):
        # only load real samples
        self.real_buffer.load(location+'_real')
    
    def add_samples(self, obs_feats, actions, next_obs_feats, rewards, terminals, sample_type='latent'):
        if sample_type == 'real':
            self.real_buffer.add_samples(obs_feats, actions, next_obs_feats, rewards, terminals)
        elif sample_type == 'latent':
            self.latent_buffer.add_samples(obs_feats, actions, next_obs_feats, rewards, terminals)
        else:
            raise NotImplementedError

    def sample(self, real_batch_size, latent_batch_size=None):
        if latent_batch_size is None:
            latent_batch_size = real_batch_size

        real_batch = self.real_buffer.sample(real_batch_size)

        data = {}

        if latent_batch_size > 0:
            latent_batch = self.latent_buffer.sample(latent_batch_size)
            for k, v in real_batch.items():
                values = np.concatenate([v, latent_batch[k]], axis=0)
                data[k] = values
        else:
            data = real_batch

        return data, real_batch_size

