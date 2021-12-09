from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.torch.sac_img import utils


class SACIMGTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            qf,
            target_qf,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            actor_update_frequency=2,
            init_temperature=0.1,
            critic_target_update_frequency=2,
            use_automatic_entropy_tuning=True,
    ):

        super().__init__()
        self.env = env
        self.policy = policy
        self.qf = qf
        self.target_qf = target_qf

        self.discount = discount
        self.reward_scale = reward_scale
        self.soft_target_tau = soft_target_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        # init the target critic with the same parameter
        self.target_qf.load_state_dict(self.qf.state_dict())
        self.policy.encoder.copy_conv_weights_from(self.qf.encoder)

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if use_automatic_entropy_tuning:
            self.log_alpha = ptu.tensor(np.log(init_temperature))
            self.log_alpha.requires_grad = True
            # set target entropy to -|A|
            self.target_entropy = -np.prod(
                    self.env.action_space.shape).item()

        # optimizers
        self.policy_optimizer = optimizer_class(self.policy.parameters(), lr=policy_lr)
        self.qf_optimizer = optimizer_class(self.qf.parameters(), lr=qf_lr)
        self.alpha_optimizer = optimizer_class([self.log_alpha], lr=policy_lr)

        # diagnostics
        self.eval_statistics = OrderedDict()

    def update_qf(self, batch):
        image_obs = batch['image_obses']
        state = batch['robot_states']
        next_image_obs = batch['next_image_obses']
        next_state = batch['next_robot_states']
        reward = batch['rewards']
        action = batch['actions']
        not_done = 1. - batch['terminals']

        with torch.no_grad():
            dist = self.policy(next_image_obs, next_state)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.target_qf(next_image_obs, next_state, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
            target_Q = self.reward_scale * reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.qf(image_obs, state, action)
        qf_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        self.eval_statistics['QF Loss'] = ptu.get_numpy(qf_loss)
        self.eval_statistics['Q Predictions'] = ptu.get_numpy(current_Q1.mean() + current_Q2.mean()) / 2.
        self.eval_statistics['Q Targets'] = ptu.get_numpy(target_Q.mean())

    def update_actor_and_alpha(self, batch):

        image_obs = batch['image_obses']
        state = batch['robot_states']

        # detach conv filters, so we don't update them with the actor loss
        dist = self.policy(image_obs, state, detach_encoder=True)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        actor_Q1, actor_Q2 = self.qf(image_obs, state, action, detach_encoder=True)
        actor_Q = torch.min(actor_Q1, actor_Q2)

        policy_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        # optimize the actor
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # log statistics
        action_mean = torch.abs(dist.mean).mean()
        action_std = dist.scale.mean()

        self.eval_statistics['Policy Loss'] = ptu.get_numpy(policy_loss)
        self.eval_statistics['Log Pis'] = ptu.get_numpy(log_prob.mean())
        self.eval_statistics['Action Scale'] = ptu.get_numpy(action_mean)
        self.eval_statistics['Action Noise'] = ptu.get_numpy(action_std)
        if self.use_automatic_entropy_tuning:
            self.eval_statistics['Alpha'] = ptu.get_numpy(self.alpha)
            self.eval_statistics['Alpha Loss'] = ptu.get_numpy(alpha_loss)

    def train_from_torch(self, batch):

        self.update_qf(batch)

        if self._num_train_steps % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(batch)

        if self._num_train_steps % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.qf, self.target_qf, self.soft_target_tau)

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self.eval_statistics = OrderedDict()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def networks(self):
        return [
            self.policy,
            self.qf,
            self.target_qf,
        ]

    @property
    def optimizers(self):
        return [
            self.alpha_optimizer,
            self.qf_optimizer,
            self.policy_optimizer,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf=self.qf,
            target_qf1=self.target_qf
        )

