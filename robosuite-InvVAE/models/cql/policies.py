
import torch
import rlkit.torch.pytorch_util as ptu
from models.img_cql.networks import AgentStep


class StMakeDeterministic(object):
    def __init__(self, distribution):
        self.distribution = distribution

    def forward(self, *args, **kwargs):
        dist = self.distribution.forward(*args, **kwargs)
        return dist.mean

    def step(self, obs, *args, **kwargs):
        robot_state = ptu.from_numpy(obs['robot_states'][None])
        object_state = ptu.from_numpy(obs['object_states'][None])

        obs = torch.cat([robot_state, object_state], dim=-1)
        action = self.forward(obs)

        assert action.ndim == 2 and action.shape[0] == 1
        agent_step = AgentStep(action=ptu.get_numpy(action), agent_info={})
        return agent_step

    def reset(self):
        pass


class VAEMakeDeterministic(object):
    def __init__(self, distribution, vae_model, use_state_input, env_mode='source'):
        self.distribution = distribution
        self.vae_model = vae_model
        self.env_mode = env_mode
        self.use_state_input = use_state_input

    def forward(self, *args, **kwargs):
        dist = self.distribution.forward(*args, **kwargs)
        return dist.mean

    def step(self, obs, *args, **kwargs):
        robot_state = ptu.from_numpy(obs['robot_states'][None])
        image_obs = ptu.from_numpy(obs['image_obses'][None]).permute(0, 3, 1, 2) / 255. - 0.5

        with torch.no_grad():
            latent = self.vae_model.encode(image_obs, self.env_mode)[0]
            input = torch.cat([robot_state, latent], dim=-1) \
                if self.use_state_input else latent
            action = self.forward(input)

        assert action.ndim == 2 and action.shape[0] == 1
        agent_step = AgentStep(action=ptu.get_numpy(action), agent_info={})
        return agent_step

    def reset(self):
        pass


