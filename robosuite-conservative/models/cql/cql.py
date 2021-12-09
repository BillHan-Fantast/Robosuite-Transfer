from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from models.img_cql.networks import AgentStep
from models.combo.combo_trainer import get_metric_avg


class CQLTrainer(TorchTrainer):
    def __init__(
            self, 
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            discount=0.99,
            reward_scale=1.0,
            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,
            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,
            use_automatic_entropy_tuning=True,
            target_update_frequency=5,
            target_entropy=None,
            temp=1.0,
            beta_penalty=1.0,
            cql_samples=10,
            **kwargs
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        self.target_update_frequency = target_update_frequency

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item() 
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class([self.log_alpha], lr=policy_lr)

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = []

        # training stats
        self._n_train_steps_total = 0
        self._policy_update_ctr = 0
        self._num_q_update_steps = 0
        self._num_policy_update_steps = 0

        # CQL params
        self.temp = temp
        self.beta_penalty = beta_penalty
        self.cql_samples = cql_samples
    
    def _get_tensor_values(self, obs, actions, network=None):
        obs_temp = obs.unsqueeze(0).repeat(actions.shape[0], 1, 1).reshape(-1, obs.shape[-1])
        preds = network(obs_temp, actions.reshape(-1, actions.shape[-1]))
        return preds.reshape(actions.shape[0], actions.shape[1], 1)

    def _get_policy_actions(self, obs, num_actions):
        obs_temp = obs.unsqueeze(0).repeat(num_actions, 1, 1)
        obs_temp = obs_temp.reshape(obs_temp.shape[0]*obs_temp.shape[1], -1)
        dist = self.policy(obs_temp)
        actions, log_pis = dist.rsample_and_logprob()
        return actions.reshape(num_actions, obs.shape[0], -1), log_pis.reshape(num_actions, obs.shape[0], 1)

    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = torch.cat([batch['robot_states'], batch['object_states']], dim=-1)
        actions = batch['actions']
        next_obs = torch.cat([batch['next_robot_states'], batch['next_object_states']], dim=-1)

        """
        Policy and Alpha Loss
        """
        dist = self.policy(obs)
        new_obs_actions, log_pi = dist.rsample_and_logprob()
        log_pi = log_pi.unsqueeze(-1)
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            alpha = self.log_alpha.exp().detach()
        else:
            alpha_loss = 0
            alpha = 1

        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions),
            self.qf2(obs, new_obs_actions),
        )
        policy_loss = (alpha * log_pi - q_new_actions).mean()
        
        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)

        with torch.no_grad():
            next_dist = self.policy(next_obs)
            next_actions, next_log_pi = next_dist.rsample_and_logprob()
            next_log_pi = next_log_pi.unsqueeze(-1)
            target_q_values = torch.min(
                self.target_qf1(next_obs, next_actions),
                self.target_qf2(next_obs, next_actions),
            ) - alpha * next_log_pi
            q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values

        qf1_loss = self.qf_criterion(q1_pred, q_target)
        qf2_loss = self.qf_criterion(q2_pred, q_target)

        random_actions_tensor = torch.FloatTensor(self.cql_samples, actions.shape[0], actions.shape[-1])\
            .uniform_(-1, 1).to(ptu.device)
        with torch.no_grad():
            curr_actions_tensor, curr_log_pis = self._get_policy_actions(obs, num_actions=self.cql_samples)
            next_actions_tensor, next_log_pis = self._get_policy_actions(next_obs, num_actions=self.cql_samples)
        q1_rand = self._get_tensor_values(obs, random_actions_tensor, network=self.qf1)
        q2_rand = self._get_tensor_values(obs, random_actions_tensor, network=self.qf2)
        q1_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, network=self.qf1)
        q2_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, network=self.qf2)
        q1_next_actions = self._get_tensor_values(obs, next_actions_tensor, network=self.qf1)
        q2_next_actions = self._get_tensor_values(obs, next_actions_tensor, network=self.qf2)

        q_std = (torch.std(q1_rand, dim=0).mean() + torch.std(q2_rand, dim=0).mean()) / 2.

        random_density = np.log(0.5 ** curr_actions_tensor.shape[-1])

        cat_q1 = torch.cat(
            [q1_rand - random_density,
             q1_next_actions - next_log_pis.detach(),
             q1_curr_actions - curr_log_pis.detach()],
            dim=0
        )
        cat_q2 = torch.cat(
            [q2_rand - random_density,
             q2_next_actions - next_log_pis.detach(),
             q2_curr_actions - curr_log_pis.detach()],
            dim=0
        )
            
        min_qf1_loss = torch.logsumexp(cat_q1 / self.temp, dim=0).mean() * self.temp
        min_qf2_loss = torch.logsumexp(cat_q2 / self.temp, dim=0).mean() * self.temp
                    
        """Subtract the log likelihood of data"""
        min_qf1_loss = min_qf1_loss - q1_pred.mean()
        min_qf2_loss = min_qf2_loss - q2_pred.mean()

        qf1_loss = qf1_loss + min_qf1_loss * self.beta_penalty
        qf2_loss = qf2_loss + min_qf2_loss * self.beta_penalty

        """
        Update networks
        """
        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self._num_policy_update_steps += 1
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self._num_q_update_steps += 1
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        """
        Soft Updates
        """
        if self._num_q_update_steps % self.target_update_frequency == 0:
            ptu.soft_update_from_to(
                self.qf1, self.target_qf1, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                self.qf2, self.target_qf2, self.soft_target_tau
            )

        metrics = OrderedDict()
        metrics['QF Loss'] = ptu.get_numpy(qf1_loss.mean())
        metrics['QF Std'] = ptu.get_numpy(q_std)
        metrics['QF ID'] = ptu.get_numpy(q1_curr_actions.mean())
        metrics['QF RAND'] = ptu.get_numpy(q1_rand.mean())
        metrics['Log Pis'] = ptu.get_numpy(log_pi.mean())
        metrics['Policy Mu'] = ptu.get_numpy(torch.abs(new_obs_actions).mean())
        if self.use_automatic_entropy_tuning:
            metrics['Alpha'] = alpha.item()
            metrics['Alpha Loss'] = alpha_loss.item()

        self._n_train_steps_total += 1
        self.eval_statistics.append(metrics)

    def get_diagnostics(self, phase='agent'):
        return get_metric_avg(self.eval_statistics)

    def end_epoch(self, epoch):
        self.eval_statistics = []

    @property
    def networks(self):
        base_list = [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]
        return base_list

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
        )


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


