
import torch
import numpy as np
from models.utils.buffer import buffer_to, numpify_buffer
from models.utils.collections import namedarraytuple
from models.utils.tensor import infer_leading_dims
from typing import Iterable
from torch.nn import Module

from models.combo.rnns import get_feat, get_dist, fn_states
from models.replay_buffer import LatentReplayBuffer

loss_info_fields = ['model_loss', 'actor_loss', 'value_loss', 'prior_entropy', 'post_entropy', 'divergence',
                    'reward_loss', 'image_loss', 'pcont_loss']
LossInfo = namedarraytuple('LossInfo', loss_info_fields)
OptInfo = namedarraytuple("OptInfo",
                          ['loss', 'grad_norm_model', 'grad_norm_actor', 'grad_norm_value'] + loss_info_fields)


class ComboTrainer(object):
    def __init__(
            self,
            agent,
            real_batch_size=50,
            fake_batch_size=50,
            model_lr=6e-4,
            qf_lr=8e-5,
            ac_lr=8e-5,
            grad_clip=100.0,
            discount=0.99,
            horizon=15,
            optim_kwargs=None,
            buffer_size=int(1e6),
            cql_samples=8,
            kl_scale=1.,
            state_scale=10.0,
            image_scale=1.0,
            reward_scale=1.0,
            free_nats=0.,
            beta_penalty=1.0,
            target_tau=0.01,
            weight_decay=0.,
            actor_update_frequency=1,
            target_update_frequency=1,
            device='cpu'
    ):
        # Algorithm's configuration
        if optim_kwargs is None:
            optim_kwargs = {}
        self.real_batch_size = real_batch_size
        self.fake_batch_size = fake_batch_size
        self.model_lr = model_lr
        self.qf_lr = qf_lr
        self.ac_lr = ac_lr
        self.grad_clip = grad_clip
        self.discount = discount
        self.horizon = horizon
        self.buffer_size = buffer_size
        self.cql_samples = cql_samples
        self.kl_scale = kl_scale
        self.state_scale = state_scale
        self.image_scale = image_scale
        self.reward_scale = reward_scale
        self.free_nats = free_nats
        self.beta_penalty = beta_penalty
        self.soft_target_tau = target_tau
        self.actor_update_frequency = actor_update_frequency
        self.target_update_frequency = target_update_frequency
        self.update_counter = 0
        self.device = device

        self.agent = agent
        self.use_robot_state = agent.use_robot_state

        # Replay Buffer
        self.latent_buffer = LatentReplayBuffer(self.buffer_size, self.buffer_size,
                                                self.agent.deterministic_size + self.agent.stochastic_size,
                                                int(np.prod(self.agent.action_shape)))

        self.model_modules = [agent.observation_encoder,
                              agent.observation_decoder,
                              agent.reward_model,
                              agent.representation]

        if self.use_robot_state:
            self.model_modules += [agent.state_decoder]

        self.actor_modules = [agent.action_model]
        self.qf_modules = [agent.qf1_model, agent.qf2_model]
        self.model_optimizer = torch.optim.Adam(get_parameters(self.model_modules), lr=self.model_lr,
                                                weight_decay=weight_decay, **optim_kwargs)
        self.actor_optimizer = torch.optim.Adam(get_parameters(self.actor_modules), lr=self.ac_lr, **optim_kwargs)
        self.qf_optimizer = torch.optim.Adam(get_parameters(self.qf_modules), lr=self.qf_lr, **optim_kwargs)

        self._agent_itrs = 0
        self._model_itrs = 0

        self._model_metrics = {}
        self._agent_metrics = {}

    def get_opt_snapshot(self):
        """Return the optimizer state dict (e.g. Adam); overwrite if using
                multiple optimizers."""
        return dict(
            model_optimizer=self.model_optimizer,
            actor_optimizer=self.actor_optimizer,
            qf_optimizer=self.qf_optimizer,
            model_modules=self.model_modules,
            actor_modules=self.actor_modules,
            qf_modules=self.qf_modules
        )

    def load_opt_snapshot(self, snapshot, load_opts):
        """Load an optimizer state dict; should expect the format returned
        from ``optim_state_dict().``"""
        if 'model_opt' in load_opts:
            self.model_optimizer = snapshot['model_optimizer']
            self.model_modules = snapshot['model_modules']
        if 'actor_opt' in load_opts:
            self.actor_optimizer = snapshot['actor_optimizer']
            self.actor_modules = snapshot['actor_modules']
        if 'qf_opt' in load_opts:
            self.qf_optimizer = snapshot['qf_optimizer']
            self.qf_modules = snapshot['qf_modules']

    def train_agent(self, steps):
        critic_metrics, actor_metrics = [], []
        for step in range(steps):
            data, real_size = self.latent_buffer.sample(self.real_batch_size,
                                                        self.fake_batch_size)
            data = buffer_to(data, self.device)
            critic_metrics.append(self.optimize_critic(data, real_size))
            if self._agent_itrs % self.actor_update_frequency == 0:
                actor_metrics.append(self.optimize_actor(data))
            if self._agent_itrs % self.target_update_frequency == 0:
                self.update_target()
            self._agent_itrs += 1
        critic_metrics = get_metric_avg(critic_metrics)
        actor_metrics = get_metric_avg(actor_metrics)

        self._agent_metrics = merge_dict(critic_metrics, actor_metrics)

    def optimize_critic(self, data, real_size):
        obs = data['obs']
        actions = data['actions']
        next_obs = data['next_obs']
        rewards = data['rewards']
        terminals = data['terminals']

        model = self.agent

        with torch.no_grad():
            next_actions = model.action_model(next_obs)
            qt_input = torch.cat([next_obs, next_actions], dim=-1)
            qt1, qt2 = model.qf1_target(qt_input), model.qf2_target(qt_input)
            qt = rewards + self.discount * (1. - terminals) * torch.minimum(qt1, qt2)
            new_actions = model.action_model(obs)

        # q predictions
        q_input = torch.cat([obs, actions], dim=-1)
        q1, q2 = model.qf1_model(q_input), model.qf2_model(q_input)

        # compute log_sum with uniform actions in [-1, 1]
        negative_actions = torch.rand((self.cql_samples,)+actions.shape).to(self.device)
        negative_actions = 2. * (negative_actions - 0.5)
        negative_actions = torch.cat([negative_actions, new_actions.unsqueeze(0)], dim=0)

        negative_obs = torch.repeat_interleave(obs.unsqueeze(0), self.cql_samples+1, dim=0)
        negative_q_input = torch.cat([negative_obs, negative_actions], dim=-1)
        neg_q1, neg_q2 = model.qf1_model(negative_q_input), model.qf2_model(negative_q_input)

        penalty_q1, penalty_q2 = torch.logsumexp(neg_q1, dim=0), torch.logsumexp(neg_q2, dim=0)

        q1_td_loss, q2_td_loss = torch.mean((q1 - qt) ** 2), torch.mean((q2 - qt) ** 2)

        q1_loss = self.beta_penalty * (penalty_q1.mean() - q1[:real_size].mean()) + q1_td_loss
        q2_loss = self.beta_penalty * (penalty_q2.mean() - q2[:real_size].mean()) + q2_td_loss
        q_loss = q1_loss + q2_loss

        self.qf_optimizer.zero_grad()
        q_loss.backward()
        self.qf_optimizer.step()

        metric = {
            'Qs/Negative Qs': neg_q1.mean().item(),
            'Qs/Target Qs': qt.mean().item(),
            'Qs/Q Loss': q_loss.item(),
            'Qs/TD Loss': q1_td_loss.item()
        }

        return metric

    def optimize_actor(self, data):
        obs = data['obs']
        model = self.agent
        actions = model.action_model(obs)
        q_input = torch.cat([obs, actions], dim=-1)
        q1, q2 = model.qf1_model(q_input), model.qf2_model(q_input)
        q = torch.minimum(q1, q2)
        actor_loss = - torch.mean(q)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        metric = {
            'Actor/Pi Abs': torch.abs(actions).mean().item(),
            'Actor/Qs': q.mean().item()
        }

        return metric

    def update_target(self):
        soft_update_from_to(
            self.agent.qf1_model, self.agent.qf1_target, self.soft_target_tau
        )
        soft_update_from_to(
            self.agent.qf2_model, self.agent.qf2_target, self.soft_target_tau
        )

    def fit_model(self, data_loader):
        metrics = []
        for i, data in enumerate(data_loader):
            data = buffer_to(data, self.device)
            stats = self.optimize_model(data, mode='train')
            metrics.append(stats)
        self._model_metrics = get_metric_avg(metrics)

    def rollout_model(self, image_obs, robot_state, action):
        model = self.agent
        lead_dim, batch_t, batch_b, img_shape = infer_leading_dims(image_obs, 3)

        # preprocess image
        image_obs = image_obs / 255.0 - 0.5
        image_obs = image_obs.permute(0, 1, 4, 2, 3)

        if self.use_robot_state:
            observation = {'image_obses': image_obs,  'robot_states': robot_state}
        else:
            observation = image_obs

        embed = model.observation_encoder(observation)
        prev_state = model.representation.initial_state(batch_b, device=self.device, dtype=action.dtype)
        prior, post = model.rollout.rollout_representation(batch_t, embed, action, prev_state)
        return prior, post, image_obs

    def optimize_model(self, data, mode='train'):
        model = self.agent

        image_obs = data['image_obses']
        robot_state = data['robot_states']
        action = data['actions']
        reward = data['rewards']

        lead_dim, batch_t, batch_b, img_shape = infer_leading_dims(image_obs, 3)
        prior, post, image_obs = self.rollout_model(image_obs, robot_state, action)

        # Model Loss
        feat = get_feat(post)
        image_pred = model.observation_decoder(feat)
        reward_pred = model.reward_model(feat)
        reward_loss = -torch.mean(reward_pred.log_prob(reward))
        image_loss = -torch.mean(image_pred.log_prob(image_obs))
        image_loss *= self.image_scale
        reward_loss *= self.reward_scale
        reward_mse = torch.sum((reward_pred.mean - reward) ** 2, dim=-1)
        image_mse = torch.sum((image_pred.mean.reshape(batch_t, batch_b, -1)
                               - image_obs.reshape(batch_t, batch_b, -1)) ** 2, dim=-1)

        state_loss, state_mse = torch.zeros_like(reward_loss), torch.zeros_like(reward_mse)
        if self.use_robot_state:
            state_pred = model.state_decoder(feat)
            state_loss = -torch.mean(state_pred.log_prob(robot_state))
            state_loss *= self.state_scale
            state_mse = torch.sum((state_pred.mean - robot_state) ** 2, dim=-1)

        prior_dist = get_dist(prior)
        post_dist = get_dist(post)
        div = torch.distributions.kl.kl_divergence(post_dist, prior_dist)
        div = torch.max(div, div.new_full(div.size(), self.free_nats)).mean()
        model_loss = self.kl_scale * div + reward_loss + image_loss + state_loss

        # latent stats
        with torch.no_grad():
            latent_mse = torch.sum((prior_dist.mean - post_dist.mean) ** 2, dim=-1)
            latent_norm = torch.sum(post_dist.mean ** 2, dim=-1)
            latent_mean = torch.mean(post_dist.mean, dim=0).mean(dim=0)
            latent_std = torch.sum((post_dist.mean - latent_mean) ** 2, dim=-1)
            latent_error_ratio = latent_mse / torch.max(latent_norm, latent_norm.new_full(latent_norm.size(), 1e-4))
            latent_std_ratio = latent_mse / torch.max(latent_std, latent_std.new_full(latent_std.size(), 1e-4))

        self.model_optimizer.zero_grad()
        model_loss.backward()
        grad_norm_model = torch.nn.utils.clip_grad_norm_(get_parameters(self.model_modules), self.grad_clip)
        if mode == 'train':
            self.model_optimizer.step()
            self._model_itrs += 1

        metric = {
            'KL div': div.item(),
            'Reward Loss': reward_loss.item(),
            'Image Loss': image_loss.item(),
            'State Loss': state_loss.item(),
            'Reward MSE': reward_mse.mean().item(),
            'Image MSE': image_mse.mean().item(),
            'State MSE': state_mse.mean().item(),
            'Latent Pred MSE': latent_mse.mean().item(),
            'Latent Pred Norm': latent_norm.mean().item(),
            'Latent Pred Std': latent_std.mean().item(),
            'Latent Pred Error Rate(Norm)': latent_error_ratio.mean().item(),
            'Latent Pred Error Rate(Std)': latent_std_ratio.mean().item(),
            'Model Grad Norm': grad_norm_model.item(),
        }

        return metric

    def visualize_model(self, data):
        """
        Visualize the Image, Reconstruction, Prior-Reconstruction
        Only visualize the first 250 steps reconstruction, and visualize the model rollouts for the next 10 steps.
        """
        model = self.agent

        with torch.no_grad():
            image_obs = data['image_obses'][:200]
            robot_state = data['robot_states'][:200]
            action = data['actions'][:200]

            prior, post, image_obs = self.rollout_model(image_obs, robot_state, action)
            img_shape = image_obs.shape[-3:]

            prior_feat = get_feat(prior)
            post_feat = get_feat(post)
            prior_recon = model.observation_decoder(prior_feat)
            post_recon = model.observation_decoder(post_feat)

            recon_imgs = torch.cat([image_obs, post_recon.mean, prior_recon.mean], dim=1) + 0.5

            get_slice = lambda state: state[10:200:50, 0]
            init_state_slice = fn_states(post, get_slice)
            action_slice, img_slice = [], []
            for step in range(10, 200, 50):
                action_slice.append(action[step+1:step+11])
                img_slice.append(image_obs[step+1:step+11])
            action_slice = torch.cat(action_slice, dim=1)
            img_slice = torch.cat(img_slice, dim=1)

            post_states = self.agent.rollout.rollout_transition(10, action_slice, init_state_slice, False)
            post_feats = get_feat(post_states)
            roll_imgs = model.observation_decoder(post_feats).mean.transpose(0, 1)
            img_slice = img_slice.transpose(0, 1)

            roll_imgs = torch.cat([img_slice.reshape(-1, *img_shape).unsqueeze(1),
                                   roll_imgs.reshape(-1, *img_shape).unsqueeze(1)], dim=1) + 0.5

            return recon_imgs.reshape(-1, *img_shape), roll_imgs.reshape(-1, *img_shape)

    def sample_data(self, num_batch, data_loader):
        for i, data in enumerate(data_loader):
            if i >= num_batch:
                return
            data = buffer_to(data, self.device)
            self.generate_latent_data(data)

    def process_episode_to_real_buffer(self, episode):

        prior, post, image_obs = self.rollout_model(episode['image_obses'],
                                                    episode['robot_states'],
                                                    episode['actions'])
        feat = get_feat(post)
        feat, action = feat[:, 0], episode['actions'][:, 0]
        self.latent_buffer.add_samples(numpify_buffer(feat)[:-1],
                                       numpify_buffer(action)[1:],
                                       numpify_buffer(feat)[1:],
                                       numpify_buffer(episode['rewards'][:, 0])[1:],
                                       numpify_buffer(episode['terminals'][:, 0])[1:],
                                       sample_type='real')

    def generate_latent_data(self, data):
        image_obs = data['image_obses']
        robot_state = data['robot_states']
        action = data['actions']

        prior, post, image_obs = self.rollout_model(image_obs, robot_state, action)
        post_fn = lambda x: x.reshape(-1, x.shape[-1])
        post = fn_states(post, post_fn)

        next_states, actions = self.agent.rollout_policy(self.horizon, post)
        init_feats, next_feats = get_feat(post), get_feat(next_states)
        feats = torch.cat([init_feats.unsqueeze(0), next_feats], dim=0)

        next_feats = feats[1:].reshape(-1, feats.shape[-1])
        feats = feats[:-1].reshape(-1, feats.shape[-1])
        actions = actions.reshape(-1, actions.shape[-1])
        rewards = self.agent.reward_model(next_feats).mean
        terminals = torch.zeros_like(rewards)

        self.latent_buffer.add_samples(numpify_buffer(feats),
                                       numpify_buffer(actions),
                                       numpify_buffer(next_feats),
                                       numpify_buffer(rewards),
                                       numpify_buffer(terminals),
                                       sample_type='latent')


    def compute_return(self,
                       reward: torch.Tensor,
                       value: torch.Tensor,
                       discount: torch.Tensor,
                       bootstrap: torch.Tensor,
                       lambda_: float):
        """
        Compute the discounted reward for a batch of data.
        reward, value, and discount are all shape [horizon - 1, batch, 1] (last element is cut off)
        Bootstrap is [batch, 1]
        """
        next_values = torch.cat([value[1:], bootstrap[None]], 0)
        target = reward + discount * next_values * (1 - lambda_)
        timesteps = list(range(reward.shape[0] - 1, -1, -1))
        outputs = []
        accumulated_reward = bootstrap
        for t in timesteps:
            inp = target[t]
            discount_factor = discount[t]
            accumulated_reward = inp + discount_factor * lambda_ * accumulated_reward
            outputs.append(accumulated_reward)
        returns = torch.flip(torch.stack(outputs), [0])
        return returns

    def model_training_phase(self):
        self.agent.train_model()
        self.agent.eval_agent()

    def agent_training_phase(self):
        self.agent.eval_model()
        self.agent.train_mode()

    def get_diagnostics(self, phase='agent'):
        if phase == 'agent':
            return self._agent_metrics
        elif phase == 'model':
            return self._model_metrics
        else:
            return NotImplementedError

    def end_epoch(self, epoch):
        pass


def get_parameters(modules: Iterable[Module]):
    """
    Given a list of torch modules, returns a list of their parameters.
    :param modules: iterable of modules
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters


def get_metric_avg(metrics):
    metric_keys = list(metrics[0].keys())
    avg_metrics = {}
    for key in metric_keys:
        values = []
        for m in metrics:
            values.append(m[key])
        avg_metrics[key] = np.mean(values)
    return avg_metrics


def merge_dict(A, B):
    O = {}
    for k, v in A.items():
        O[k] = v
    for k, v in B.items():
        O[k] = v
    return O


def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )
