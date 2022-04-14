import numpy as np
import torch
import torch.nn as nn
from models.utils.collections import namedarraytuple

from models.combo.observation import ImageDecoder, ImageEncoder, ProprioImageEncoder, conv_out_shape
from models.combo.rnns import RSSMState, RSSMRepresentation, RSSMTransition, RSSMRollout

ModelReturnSpec = namedarraytuple('ModelReturnSpec', ['action', 'state'])
DreamerAgentInfo = namedarraytuple('DreamerAgentInfo', ['prev_state'])
AgentStep = namedarraytuple("AgentStep", ["action", "agent_info"])


class RSSMModel(nn.Module):
    def __init__(
            self,
            action_shape,
            image_shape,
            stochastic_size=30,
            deterministic_size=200,
            model_hidden=200,
            num_models=1,
            conv_depth=32,
            **kwargs
    ):
        super().__init__()

        # Image
        self.observation_encoder = ImageEncoder(shape=image_shape, depth=conv_depth)
        encoder_embed_size = self.observation_encoder.embed_size
        decoder_embed_size = stochastic_size + deterministic_size
        self.observation_decoder = ImageDecoder(embed_size=decoder_embed_size, shape=image_shape, depth=conv_depth)

        # Agent
        self.action_shape = action_shape
        output_size = np.prod(action_shape)
        self.feature_size = stochastic_size + deterministic_size

        # Model
        self.transition = RSSMTransition(output_size, num_models, stochastic_size, deterministic_size, model_hidden)
        self.representation = RSSMRepresentation(self.transition, encoder_embed_size, output_size,
                                                 stochastic_size, deterministic_size, model_hidden)
        self.rollout = RSSMRollout(self.representation, self.transition)

        # Qs
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size


class InverseModel(nn.Module):
    def __init__(
            self,
            action_shape,
            image_shape,
            hidden_size=1024,
            depth=32,
            stride=2,
            activation=nn.ReLU,
            **kwargs
    ):
        super().__init__()

        self.convolutions = nn.Sequential(
            nn.Conv2d(image_shape[2], 1 * depth, 4, stride),
            activation(),
            nn.Conv2d(1 * depth, 2 * depth, 4, stride),
            nn.BatchNorm2d(2 * depth),
            activation(),
            nn.Conv2d(2 * depth, 4 * depth, 4, stride),
            activation(),
            nn.Conv2d(4 * depth, 8 * depth, 4, stride),
            activation(),
            nn.Conv2d(8 * depth, 16 * depth, 3, stride),
            nn.BatchNorm2d(16 * depth),
            activation()
        )

        conv1_shape = conv_out_shape(image_shape[:-1], 0, 4, stride)
        conv2_shape = conv_out_shape(conv1_shape, 0, 4, stride)
        conv3_shape = conv_out_shape(conv2_shape, 0, 4, stride)
        conv4_shape = conv_out_shape(conv3_shape, 0, 4, stride)
        conv5_shape = conv_out_shape(conv4_shape, 0, 3, stride)
        conv_out_size = 16 * depth * np.prod(conv5_shape).item()
        embed_size = 2 * conv_out_size

        self.networks = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, action_shape[0])
        )

    def forward(self, obs, n_obs):
        obs = self.convolutions(obs).reshape(obs.shape[0], -1)
        n_obs = self.convolutions(n_obs).reshape(n_obs.shape[0], -1)
        input = torch.cat([obs, n_obs], dim=-1)
        return torch.tanh(self.networks(input))
