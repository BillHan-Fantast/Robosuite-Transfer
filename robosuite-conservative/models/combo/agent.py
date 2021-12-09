import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from models.utils.buffer import buffer_func, buffer_to, numpify_buffer
from models.utils.collections import namedarraytuple

from models.combo.action import ActionNetwork
from models.combo.dense import DenseModel
from models.combo.observation import ImageDecoder, ImageEncoder, ProprioImageEncoder
from models.combo.rnns import RSSMState, RSSMRepresentation, RSSMTransition, RSSMRollout, get_feat

ModelReturnSpec = namedarraytuple('ModelReturnSpec', ['action', 'state'])
DreamerAgentInfo = namedarraytuple('DreamerAgentInfo', ['prev_state'])
AgentStep = namedarraytuple("AgentStep", ["action", "agent_info"])


class AgentModel(nn.Module):
    def __init__(
            self,
            action_shape=(5,),
            stochastic_size=30,
            deterministic_size=200,
            model_hidden=200,
            image_shape=(3, 64, 64),
            state_shape=(10,),
            state_hidden=200,
            state_layers=3,
            action_hidden=200,
            action_layers=3,
            action_dist='none',
            reward_shape=(1,),
            reward_layers=3,
            reward_hidden=300,
            value_shape=(1,),
            value_layers=3,
            value_hidden=200,
            num_models=1,
            conv_depth=32,
            device='cpu',
            use_robot_state=False,
            **kwargs
    ):
        super().__init__()

        # Image
        if use_robot_state:
            self.observation_encoder = ProprioImageEncoder(img_shape=image_shape,
                                                           state_shape=state_shape,
                                                           depth=conv_depth)
        else:
            self.observation_encoder = ImageEncoder(shape=image_shape, depth=conv_depth)

        encoder_embed_size = self.observation_encoder.embed_size
        decoder_embed_size = stochastic_size + deterministic_size

        self.observation_decoder = ImageDecoder(embed_size=decoder_embed_size, shape=image_shape, depth=conv_depth)

        # Agent
        self.action_shape = action_shape
        output_size = np.prod(action_shape)
        feature_size = stochastic_size + deterministic_size
        self.action_size = output_size
        self.action_dist = action_dist
        self.action_model = ActionNetwork(output_size, feature_size, action_hidden, action_layers)

        # Model
        self.transition = RSSMTransition(output_size, num_models, stochastic_size, deterministic_size, model_hidden)
        self.representation = RSSMRepresentation(self.transition, encoder_embed_size, output_size, stochastic_size,
                                                 deterministic_size, model_hidden)
        self.rollout = RSSMRollout(self.representation, self.transition)

        # Decoder
        self.reward_model = DenseModel(feature_size, reward_shape, reward_layers, reward_hidden)
        self.state_decoder = DenseModel(feature_size, state_shape, state_layers, state_hidden)

        # Qs
        self.qf1_model = DenseModel(feature_size+output_size, value_shape, value_layers, value_hidden, 'none')
        self.qf2_model = DenseModel(feature_size+output_size, value_shape, value_layers, value_hidden, 'none')
        self.qf1_target = deepcopy(self.qf1_model)
        self.qf2_target = deepcopy(self.qf2_model)
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size
        self.use_robot_state = use_robot_state

        # set mode, gpu
        self._model_mode = 'train'
        self._actor_mode = 'train'
        self.dtype = torch.float32
        self.device = device

    def policy(self, state: RSSMState):
        feat = get_feat(state)
        action_dist = self.action_model(feat)
        if self.action_dist == 'tanh_normal':
            if self.training:  # use agent.train(bool) or agent.eval()
                action = action_dist.rsample()
            else:
                action = action_dist.mode()
        elif self.action_dist == 'one_hot':
            action = action_dist.sample()
            # This doesn't change the value, but gives us straight-through gradients
            action = action + action_dist.probs - action_dist.probs.detach()
        elif self.action_dist == 'relaxed_one_hot':
            action = action_dist.rsample()
        elif self.action_dist == 'none':
            action = action_dist
        else:
            action = action_dist.sample()
        return action, action_dist

    def get_state_representation(self, observation, prev_action, prev_state: RSSMState = None):
        """

        :param observation: size(batch, channels, width, height)
        :param prev_action: size(batch, action_size)
        :param prev_state: RSSMState: size(batch, state_size)
        :return: RSSMState
        """
        obs_embed = self.observation_encoder(observation)
        if prev_state is None:
            prev_state = self.representation.initial_state(prev_action.shape[0], device=prev_action.device,
                                                           dtype=prev_action.dtype)
        _, state = self.representation(obs_embed, prev_action, prev_state)
        return state

    def get_state_transition(self, prev_action: torch.Tensor, prev_state: RSSMState):
        """

        :param prev_action: size(batch, action_size)
        :param prev_state: RSSMState: size(batch, state_size)
        :return: RSSMState
        """
        state = self.transition(prev_action, prev_state)
        return state

    def get_model_snapshot(self):
        return dict(
            observation_encoder=self.observation_encoder,
            observation_decoder=self.observation_decoder,
            reward_model=self.reward_model,
            state_decoder=self.state_decoder,
            transition=self.transition,
            representation=self.representation,
            rollout=self.rollout
        )

    def load_model_snapshot(self, snapshot):
        self.observation_encoder = snapshot['observation_encoder']
        self.observation_decoder = snapshot['observation_decoder']
        self.reward_model = snapshot['reward_model']
        self.state_decoder = snapshot['state_decoder']
        self.transition = snapshot['transition']
        self.representation = snapshot['representation']
        self.rollout = snapshot['rollout']

    def train_model(self):
        self.representation.train()
        self.observation_encoder.train()
        self.observation_decoder.train()
        self.reward_model.train()
        if self.use_robot_state:
            self.state_decoder.train()
        self._model_mode = 'train'

    def eval_model(self):
        self.representation.eval()
        self.observation_encoder.eval()
        self.observation_decoder.eval()
        self.reward_model.eval()
        if self.use_robot_state:
            self.state_decoder.eval()
        self._model_mode = 'eval'

    def train_agent(self):
        """Go into training mode (e.g. see PyTorch's ``Module.train()``)."""
        self.qf1_model.train()
        self.qf2_model.train()
        self.action_model.train()
        self._actor_mode = 'train'

    def eval_agent(self):
        """Go into evaluation mode.  Example use could be to adjust epsilon-greedy."""
        self.qf1_model.eval()
        self.qf2_model.eval()
        self.action_model.eval()
        self._actor_mode = 'eval'


class ComboAgent(AgentModel):

    def __init__(self, expl_amount=0.4, eval_noise=0, expl_type="additive_gaussian",
                 expl_min=0.2, expl_decay=7000, **kwargs):

        self.expl_amount = expl_amount
        self.eval_noise = eval_noise
        self.expl_type = expl_type
        self.expl_min = expl_min
        self.expl_decay = expl_decay
        super().__init__(**kwargs)
        self._itr = 0
        self._mode = 'train'
        self._prev_rnn_state = None

    @torch.no_grad()
    def step(self, observation, prev_action):
        """"
        Compute policy's action distribution from inputs, and sample an
        action. Calls the model to produce mean, log_std, value estimate, and
        next recurrent state.  Moves inputs to device and returns outputs back
        to CPU, for the sampler.  Advances the recurrent state of the agent.
        (no grad)
        """

        model_inputs = buffer_to((observation['image_obses'], observation['robot_states'],
                                  prev_action), device=self.device)
        action, state = self.forward(*model_inputs, self.prev_rnn_state)
        action = self.exploration(action)
        # Model handles None, but Buffer does not, make zeros if needed:
        prev_state = self.prev_rnn_state or buffer_func(state, torch.zeros_like)
        self.advance_rnn_state(state)
        agent_info = DreamerAgentInfo(prev_state=prev_state)
        agent_step = AgentStep(action=action, agent_info=agent_info)
        return numpify_buffer(agent_step)

    def exploration(self, action: torch.Tensor) -> torch.Tensor:
        """
        :param action: action to take, shape (1,) (if categorical), or (action dim,) (if continuous)
        :return: action of the same shape passed in, augmented with some noise
        """
        if self._mode in ['train', 'sample']:
            expl_amount = self.expl_amount
            if self.expl_decay:  # Linear decay
                expl_amount = expl_amount - self._itr / self.expl_decay
            if self.expl_min:
                expl_amount = max(self.expl_min, expl_amount)
        elif self._mode == 'eval':
            expl_amount = self.eval_noise
        else:
            raise NotImplementedError

        if self.expl_type == 'additive_gaussian':  # For continuous actions
            noise = torch.randn(action.shape, device=action.device) * expl_amount
            return torch.clamp(action + noise, -1, 1)
        if self.expl_type == 'completely_random':  # For continuous actions
            if expl_amount == 0:
                return action
            else:
                return torch.rand(action.shape, device=action.device) * 2 - 1.  # scale to [-1, 1]
        if self.expl_type == 'epsilon_greedy':  # For discrete actions
            action_dim = self.env_model_kwargs["action_shape"][0]
            if np.random.uniform(0, 1) < expl_amount:
                index = torch.randint(0, action_dim, action.shape[:-1], device=action.device)
                action = torch.zeros_like(action)
                action[..., index] = 1
            return action
        raise NotImplementedError(self.expl_type)

    def rollout_policy(self, steps, init_state: RSSMState):
        policy = lambda state: self.exploration(self.policy(state)[0])
        return self.rollout.rollout_policy(steps, policy, init_state)

    def forward(self, image_obs, robot_state, prev_action, prev_state: RSSMState = None):
        image_obs = image_obs.reshape(1, *image_obs.shape).to(self.dtype) / 255.0 - 0.5
        image_obs = image_obs.permute(0, 3, 1, 2)
        robot_state = robot_state.reshape(1, *robot_state.shape).to(self.dtype)
        prev_action = prev_action.reshape(1, *prev_action.shape).to(self.dtype)

        assert len(image_obs.shape) == 4 and len(robot_state.shape) == 2 and len(prev_action.shape) == 2

        if self.use_robot_state:
            observation = {'image_obses': image_obs, 'robot_states': robot_state}
        else:
            observation = image_obs

        if prev_state is None:
            prev_state = self.representation.initial_state(prev_action.shape[0], device=self.device, dtype=self.dtype)

        state = self.get_state_representation(observation, prev_action, prev_state)

        action, action_dist = self.policy(state)
        return_spec = ModelReturnSpec(action, state)
        return return_spec

    def reset(self):
        """Sets the recurrent state to ``None``, which built-in PyTorch
        modules conver to zeros."""
        self._prev_rnn_state = None

    def advance_rnn_state(self, new_rnn_state):
        """Sets the recurrent state to the newly computed one (i.e. recurrent agents should
        call this at the end of their ``step()``). """
        self._prev_rnn_state = new_rnn_state

    @property
    def prev_rnn_state(self):
        return self._prev_rnn_state

    def train_mode(self):
        self._mode = 'train'
        super().train_agent()

    def sample_mode(self):
        self._mode = 'sample'
        super().eval_agent()

    def eval_mode(self):
        self._mode = 'eval'
        self._prev_rnn_state = None
        super().eval_agent()

