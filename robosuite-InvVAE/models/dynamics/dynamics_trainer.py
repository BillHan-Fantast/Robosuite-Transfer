
import torch
import numpy as np
from models.utils.buffer import buffer_to
from models.utils.tensor import infer_leading_dims

from models.combo.rnns import get_feat, get_dist
from models.combo.combo_trainer import get_metric_avg

import rlkit.torch.pytorch_util as ptu


class DynTrainer(object):
    def __init__(
            self,
            model,
            forward_lr=6e-4,
            inverse_lr=3e-4,
            grad_clip=100.0,
            kl_scale=1.,
            state_scale=10.0,
            image_scale=1.0,
            free_nats=0.,
            weight_decay=0.
    ):
        self.forward_lr = forward_lr
        self.inverse_lr = inverse_lr
        self.grad_clip = grad_clip
        self.kl_scale = kl_scale
        self.state_scale = state_scale
        self.image_scale = image_scale
        self.free_nats = free_nats

        self.model = model

        self.forward_model_optimizer = torch.optim.Adam(list(model.forward_model.parameters()),
                                                        lr=self.forward_lr, weight_decay=weight_decay)
        self.inverse_model_optimizer = torch.optim.Adam(list(model.inverse_model.parameters()),
                                                        lr=self.inverse_lr, weight_decay=weight_decay)
        self._forward_train_metrics = []
        self._inverse_train_metrics = []
        self._eval_metrics = []

    def train_model(self, source_loader, target_loader, mode):
        len_loader = min(len(source_loader), len(target_loader))
        source_loader = iter(source_loader)
        target_loader = iter(target_loader)

        for i in range(len_loader):
            source_batch = next(source_loader)
            target_batch = next(target_loader)
            source_batch = buffer_to(source_batch, ptu.device)
            target_batch = buffer_to(target_batch, ptu.device)
            self.model.train_mode('dynamics_model')
            if mode == 'forward':
                self._update_forward_model(source_batch, target_batch)
            elif mode == 'inverse':
                self._update_inverse_model(source_batch, target_batch)
            else:
                raise NotImplementedError
            self.model.eval_mode('dynamics_model')

    def eval_model(self, source_loader, target_loader):
        self.model.eval_mode('dynamics_model')

        len_loader = min(len(source_loader), len(target_loader))
        source_loader = iter(source_loader)
        target_loader = iter(target_loader)

        for i in range(len_loader):
            source_batch = next(source_loader)
            target_batch = next(target_loader)
            source_batch = buffer_to(source_batch, ptu.device)
            target_batch = buffer_to(target_batch, ptu.device)

            src_obs = source_batch['image_obses'].permute(0, 3, 1, 2) / 255. - 0.5
            trg_obs = target_batch['image_obses'].permute(0, 3, 1, 2) / 255. - 0.5
            src_ac = source_batch['actions']
            trg_ac = target_batch['actions']
            n_src_obs = source_batch['next_image_obses'].permute(0, 3, 1, 2) / 255. - 0.5
            n_trg_obs = target_batch['next_image_obses'].permute(0, 3, 1, 2) / 255. - 0.5

            pred_src_ac = self.model.inverse_model['source'](src_obs, n_src_obs)
            pred_trg_ac = self.model.inverse_model['target'](trg_obs, n_trg_obs)

            cat_src_obs = torch.stack([src_obs,  n_src_obs], dim=0)
            cat_trg_obs = torch.stack([trg_obs,  n_trg_obs], dim=0)
            cat_src_ac = torch.stack([torch.zeros_like(src_ac), src_ac], dim=0)
            cat_trg_ac = torch.stack([torch.zeros_like(trg_ac), trg_ac], dim=0)

            src_prior, _ = self.model.rollout(self.model.forward_model['source'], cat_src_obs,
                                              cat_src_ac, sample=False)
            trg_prior, _ = self.model.rollout(self.model.forward_model['target'], cat_trg_obs,
                                              cat_trg_ac, sample=False)
            pred_src_obs = self.model.forward_model['source'].observation_decoder(get_feat(src_prior)[1]).mean
            pred_trg_obs = self.model.forward_model['target'].observation_decoder(get_feat(trg_prior)[1]).mean

            pred_src_obs = pred_src_obs.reshape(pred_src_obs.shape[0], -1)
            pred_trg_obs = pred_trg_obs.reshape(pred_trg_obs.shape[0], -1)

            action_mse = torch.sum((pred_src_ac - src_ac) ** 2, dim=-1).mean() + \
                         torch.sum((pred_trg_ac - trg_ac) ** 2, dim=-1).mean()

            obs_mse = torch.sum((pred_src_obs - src_obs.reshape(src_obs.shape[0], -1)) ** 2, dim=-1).mean() + \
                      torch.sum((pred_trg_obs - trg_obs.reshape(trg_obs.shape[0], -1)) ** 2, dim=-1).mean()

            self._eval_metrics.append({
                'eval/Inverse MSE': action_mse.item() / 2.,
                'eval/Forward MSE': obs_mse.item() / 2.
            })

    def _update_forward_model(self, source_data, target_data):
        model_loss = torch.FloatTensor([0.]).to(ptu.device)

        for domain, data in zip(['source', 'target'], [source_data, target_data]):
            model = self.model.forward_model[domain]
            image_obs = data['image_obses'].permute(0, 1, 4, 2, 3) / 255. - 0.5
            action = data['actions']

            lead_dim, batch_t, batch_b, img_shape = infer_leading_dims(image_obs, 3)
            prior, post = self.model.rollout(model, image_obs, action, sample=True)

            # Model Loss
            feat = get_feat(post)
            image_pred = model.observation_decoder(feat)
            image_loss = -self.image_scale * torch.mean(image_pred.log_prob(image_obs))
            image_mse = torch.sum((image_pred.mean.reshape(batch_t, batch_b, -1) -
                                   image_obs.reshape(batch_t, batch_b, -1)) ** 2, dim=-1)

            prior_dist = get_dist(prior)
            post_dist = get_dist(post)
            div = torch.distributions.kl.kl_divergence(post_dist, prior_dist)
            div = torch.max(div, div.new_full(div.size(), self.free_nats)).mean()
            model_loss += self.kl_scale * div + image_loss

            # latent stats
            with torch.no_grad():
                latent_mse = torch.sum((prior_dist.mean - post_dist.mean) ** 2, dim=-1)
                latent_norm = torch.sum(post_dist.mean ** 2, dim=-1)
                latent_mean = torch.mean(post_dist.mean, dim=0).mean(dim=0)
                latent_std = torch.sum((post_dist.mean - latent_mean) ** 2, dim=-1)
                latent_std_ratio = latent_mse / torch.max(latent_std, latent_std.new_full(latent_std.size(), 1e-4))

            metric = {
                'train/KL div': div.item(),
                'train/Image Loss': image_loss.item(),
                'train/Image MSE': image_mse.mean().item(),
                'train/Latent Pred MSE': latent_mse.mean().item(),
                'train/Latent Pred Norm': latent_norm.mean().item(),
                'train/Latent Pred Std': latent_std.mean().item(),
                'train/Latent Error Rate': latent_std_ratio.mean().item()
                }

            self._forward_train_metrics.append(metric)

        self.forward_model_optimizer.zero_grad()
        model_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.model.forward_model['source'].parameters()), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(list(self.model.forward_model['target'].parameters()), self.grad_clip)
        self.forward_model_optimizer.step()

    def _update_inverse_model(self, source_data, target_data):
        model_loss = torch.FloatTensor([0.]).to(ptu.device)

        for domain, data in zip(['source', 'target'], [source_data, target_data]):
            model = self.model.inverse_model[domain]
            obs = data['image_obses'].permute(0, 3, 1, 2) / 255. - 0.5
            next_obs = data['next_image_obses'].permute(0, 3, 1, 2) / 255. - 0.5
            action = data['actions']
            action_pred = model(obs, next_obs)
            action_mse = torch.sum((action_pred - action) ** 2, dim=-1).mean()
            model_loss += action_mse

            metric = {
                'train/Action Loss': action_mse.item()
            }

            self._inverse_train_metrics.append(metric)

        self.inverse_model_optimizer.zero_grad()
        model_loss.backward()
        self.inverse_model_optimizer.step()

    def sample_data(self, batch, domain):
        model = self.model.forward_model[domain]

        with torch.no_grad():
            image_obs = batch['image_obses'][:10].permute(0, 3, 1, 2) / 255. - 0.5
            next_image_obs = batch['next_image_obses'][:10].permute(0, 3, 1, 2) / 255. - 0.5
            action = batch['actions'][:10]
            cat_obs = torch.stack([image_obs, next_image_obs], dim=0)
            cat_action = torch.stack([torch.zeros_like(action), action], dim=0)
            prior, post = self.model.rollout(model, cat_obs, cat_action, sample=False)
            img_shape = image_obs.shape[-3:]

            prior_feat = get_feat(prior)
            post_feat = get_feat(post)
            prior_recon = model.observation_decoder(prior_feat)
            post_recon = model.observation_decoder(post_feat)

            recon_imgs = torch.stack([next_image_obs, post_recon.mean[1], prior_recon.mean[1]], dim=1) + 0.5

            return recon_imgs.reshape(-1, *img_shape)

    def get_diagnostics(self, phase='agent'):
        metrics = {}
        if len(self._forward_train_metrics) > 0:
            metrics.update(get_metric_avg(self._forward_train_metrics))
        if len(self._inverse_train_metrics) > 0:
            metrics.update(get_metric_avg(self._inverse_train_metrics))
        if len(self._eval_metrics) > 0:
            metrics.update(get_metric_avg(self._eval_metrics))

        return metrics

    def end_epoch(self, epoch):
        self._forward_train_metrics = []
        self._inverse_train_metrics = []
        self._eval_metrics = []
