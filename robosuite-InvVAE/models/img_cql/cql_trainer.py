
import numpy as np
import torch
import torch.optim as optim

import rlkit.torch.pytorch_util as ptu
from models.utils.buffer import buffer_to
from models.combo.combo_trainer import get_metric_avg, merge_dict
from models.img_cql import utils


class IMGCQLTrainer(object):
    def __init__(
            self,
            action_shape,
            policy,
            qf,
            target_qf,
            discount=0.99,
            reward_scale=1.0,
            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,
            cql_samples=1,
            beta_penalty=1.,
            soft_target_tau=1e-2,
            actor_update_frequency=2,
            init_temperature=0.1,
            critic_target_update_frequency=2,
            use_automatic_entropy_tuning=True,
            **kwargs
    ):

        self.policy = policy
        self.qf = qf
        self.target_qf = target_qf

        self.discount = discount
        self.cql_samples = cql_samples
        self.beta_penalty = beta_penalty
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
            self.target_entropy = -np.prod(action_shape).item()

        # optimizers
        self.policy_optimizer = optimizer_class(self.policy.parameters(), lr=policy_lr)
        self.qf_optimizer = optimizer_class(self.qf.parameters(), lr=qf_lr)
        self.alpha_optimizer = optimizer_class([self.log_alpha], lr=policy_lr)

        # diagnostics
        self._agent_itrs = 0
        self._agent_metrics = {}

    def update_qf(self, batch):
        image_obs = batch['image_obses']
        state = batch['robot_states']
        next_image_obs = batch['next_image_obses']
        next_state = batch['next_robot_states']
        reward = batch['rewards']
        action = batch['actions']
        not_done = 1. - batch['terminals']

        image_obs = image_obs.permute(0, 3, 1, 2) / 255. - 0.5
        next_image_obs = next_image_obs.permute(0, 3, 1, 2) / 255. - 0.5

        with torch.no_grad():
            dist = self.policy(next_image_obs, next_state)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            qt1, qt2 = self.target_qf(next_image_obs, next_state, next_action)
            vt = torch.min(qt1, qt2) - self.alpha.detach() * log_prob
            qt = self.reward_scale * reward + (not_done * self.discount * vt)
            new_action = self.policy(image_obs, state).rsample()

        # get current Q estimates
        q1, q2 = self.qf(image_obs, state, action)

        negative_action = torch.rand((self.cql_samples,)+action.shape).to(ptu.device)
        negative_action = 2. * (negative_action - 0.5)
        negative_action = torch.cat([negative_action, new_action.unsqueeze(0)], dim=0)

        negative_image_obs = image_obs.unsqueeze(0)
        negative_state = state.unsqueeze(0)
        neg_q1, neg_q2 = self.qf(negative_image_obs, negative_state, negative_action)
        penalty_q1, penalty_q2 = torch.logsumexp(neg_q1, dim=0), torch.logsumexp(neg_q2, dim=0)

        q1_td_loss, q2_td_loss = torch.mean((q1 - qt) ** 2), torch.mean((q2 - qt) ** 2)

        q1_loss = self.beta_penalty * (penalty_q1.mean() - q1.mean()) + q1_td_loss
        q2_loss = self.beta_penalty * (penalty_q2.mean() - q2.mean()) + q2_td_loss
        qf_loss = q1_loss + q2_loss

        # Optimize the critic
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        metrics = {}

        metrics['Qs/TD Loss'] = ptu.get_numpy(q1_td_loss)
        metrics['Qs/Q Preds'] = ptu.get_numpy(q1.mean() + q2.mean()) / 2.
        metrics['Qs/Target Qs'] = ptu.get_numpy(qt.mean())
        metrics['Qs/Negative Qs'] = ptu.get_numpy(neg_q1.mean())

        return metrics

    def update_actor_and_alpha(self, batch):

        image_obs = batch['image_obses']
        state = batch['robot_states']

        image_obs = image_obs.permute(0, 3, 1, 2) / 255. - 0.5

        # detach conv filters, so we don't update them with the actor loss
        dist = self.policy(image_obs, state, detach_encoder=True)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        q1, q2 = self.qf(image_obs, state, action, detach_encoder=True)
        q = torch.min(q1, q2)

        policy_loss = (self.alpha.detach() * log_prob - q).mean()

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

        metrics = {}
        metrics['Actor/Policy Loss'] = ptu.get_numpy(policy_loss)
        metrics['Actor/Log Pis'] = ptu.get_numpy(log_prob.mean())
        metrics['Actor/Pi Abs'] = ptu.get_numpy(action_mean)
        metrics['Actor/Pi Noise'] = ptu.get_numpy(action_std)
        metrics['Actor/Qs'] = ptu.get_numpy(q.mean())
        if self.use_automatic_entropy_tuning:
            metrics['Actor/Alpha'] = ptu.get_numpy(self.alpha)
            metrics['Actor/Alpha Loss'] = ptu.get_numpy(alpha_loss)

        return metrics

    def train_agent(self, steps, data_loader, data_func):
        critic_metrics, actor_metrics = [], []
        for step in range(steps):
            for i, data in enumerate(data_loader):
                data = buffer_to(data, ptu.device)
                data = data_func(data)
                critic_metrics.append(self.update_qf(data))

                if self._agent_itrs % self.actor_update_frequency == 0:
                    actor_metrics.append(self.update_actor_and_alpha(data))

                if self._agent_itrs % self.critic_target_update_frequency == 0:
                    utils.soft_update_params(self.qf, self.target_qf, self.soft_target_tau)

                self._agent_itrs += 1

        critic_metrics = get_metric_avg(critic_metrics)
        actor_metrics = get_metric_avg(actor_metrics)

        self._agent_metrics = merge_dict(critic_metrics, actor_metrics)

    def get_diagnostics(self, phase='agent'):
        if phase == 'agent':
            stats = {'Train Itrs': self._agent_itrs}
            stats.update(self._agent_metrics)
        else:
            raise NotImplementedError
        return stats

    def end_epoch(self, epoch):
        self._agent_metrics = {}

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

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)
