
import torch
from torch import nn as nn
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.sac_img import utils
from models.utils.collections import namedarraytuple

AgentStep = namedarraytuple("AgentStep", ["action", "agent_info"])


class Encoder(nn.Module):
    """Convolutional encoder for image-based observations."""
    def __init__(self, obs_shape, feature_dim):
        super().__init__()

        assert len(obs_shape) == 3 and obs_shape[0] == obs_shape[1]

        self.output_logits = False
        self.feature_dim = feature_dim

        self.num_layers = 4
        self.num_filters = 32
        self.obs_shape = obs_shape

        conv1_shape = conv_out_shape(obs_shape[:-1], 0, 3, 2)
        conv2_shape = conv_out_shape(conv1_shape, 0, 3, 1)
        conv3_shape = conv_out_shape(conv2_shape, 0, 3, 1)
        conv4_shape = conv_out_shape(conv3_shape, 0, 3, 1)

        self.output_dim = conv4_shape[0]

        self.convs = nn.ModuleList([
            nn.Conv2d(obs_shape[2], self.num_filters, 3, stride=2),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1)
        ])

        self.head = nn.Sequential(
            nn.Linear(self.num_filters * self.output_dim * self.output_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim))

        self.outputs = dict()

    def forward_conv(self, obs):

        batch_shape = obs.shape[:-3]
        obs = obs.reshape(-1, *obs.shape[-3:])

        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.reshape(*batch_shape, -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        out = self.head(h)
        if not self.output_logits:
            out = torch.tanh(out)

        self.outputs['out'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        for i in range(self.num_layers):
            utils.tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_encoder/{k}_hist', v, step)
            if len(v.shape) > 2:
                logger.log_image(f'train_encoder/{k}_img', v[0], step)

        for i in range(self.num_layers):
            logger.log_param(f'train_encoder/conv{i + 1}', self.convs[i], step)


class Actor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, image_obs_shape, state_shape, action_shape, feature_dim, hidden_dim,
                 hidden_depth, log_std_bounds, use_state_input):

        super().__init__()

        self.encoder = Encoder(image_obs_shape, feature_dim)

        latent_dim = self.encoder.feature_dim
        if use_state_input:
            latent_dim += state_shape[0]

        self.log_std_bounds = log_std_bounds
        self.trunk = utils.mlp(latent_dim, hidden_dim, 2 * action_shape[0], hidden_depth)

        self.outputs = dict()
        self.use_state_input = use_state_input
        self.apply(utils.weight_init)

    def forward(self, obs, state, detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)

        if self.use_state_input:
            obs = torch.cat([obs, state], dim=-1)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = utils.SquashedNormal(mu, std)
        return dist

    def step(self, obs, *args, **kwargs):
        image_obs = ptu.from_numpy(obs['image_obses'][None])
        robot_state = ptu.from_numpy(obs['robot_states'][None])

        image_obs = image_obs.permute(0, 3, 1, 2) / 255. - 0.5

        with torch.no_grad():
            action = self.forward(image_obs, robot_state).sample()

        assert action.ndim == 2 and action.shape[0] == 1
        agent_step = AgentStep(action=ptu.get_numpy(action), agent_info={})
        return agent_step

    def reset(self):
        pass

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)


class MakeDeterministic(object):
    def __init__(self, distribution):
        self.distribution = distribution

    def forward(self, *args, **kwargs):
        dist = self.distribution.forward(*args, **kwargs)
        return dist.mean

    def step(self, obs, *args, **kwargs):
        image_obs = ptu.from_numpy(obs['image_obses'][None])
        robot_state = ptu.from_numpy(obs['robot_states'][None])

        image_obs = image_obs.permute(0, 3, 1, 2) / 255. - 0.5

        with torch.no_grad():
            action = self.forward(image_obs, robot_state)

        assert action.ndim == 2 and action.shape[0] == 1
        agent_step = AgentStep(action=ptu.get_numpy(action), agent_info={})
        return agent_step

    def reset(self):
        pass


class Critic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, image_obs_shape, state_shape, action_shape, feature_dim,
                 hidden_dim, hidden_depth, use_state_input):
        super().__init__()

        self.encoder = Encoder(image_obs_shape, feature_dim)

        latent_dim = self.encoder.feature_dim + action_shape[0]
        if use_state_input:
            latent_dim += state_shape[0]

        self.Q1 = utils.mlp(latent_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(latent_dim, hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.use_state_input = use_state_input
        self.apply(utils.weight_init)

    def forward(self, obs, state, action, detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)

        if not obs.shape[0] == action.shape[0]:
            assert obs.shape[0] == 1
            obs = torch.repeat_interleave(obs, action.shape[0], dim=0)

        if not state.shape[0] == action.shape[0]:
            assert state.shape[0] == 1
            state = torch.repeat_interleave(state, action.shape[0], dim=0)

        if self.use_state_input:
            obs_action = torch.cat([obs, state, action], dim=-1)
        else:
            obs_action = torch.cat([obs, action], dim=-1)

        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        self.encoder.log(logger, step)

        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)

                logger.log_param(f'train_critic/q2_fc{i}', m2, step)


def conv_out(h_in, padding, kernel_size, stride):
    return int((h_in + 2. * padding - (kernel_size - 1.) - 1.) / stride + 1.)

def conv_out_shape(h_in, padding, kernel_size, stride):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)