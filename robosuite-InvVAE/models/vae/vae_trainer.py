
import torch
from torch import optim

from rlkit.torch import pytorch_util as ptu
from models.combo.combo_trainer import buffer_to, get_metric_avg


class ConvVAETrainer(object):
    def __init__(
            self,
            model,
            vae_lr,
            dis_lr,
            beta=0.5,
            c_domain=1.0,
            c_cycle=1.0,
            c_latent=1.0,
            c_paired=0.0,
            c_src_forward=0.0,
            c_src_inverse=0.0,
            c_trg_forward=0.0,
            c_trg_inverse=0.0,
            weight_decay=0,
            reparam_for_losses=False,
            num_classifier_steps_per_vae_update=1,
    ):

        self.model = model
        self.representation_size = model.representation_size

        self.vae_optim = optim.Adam(list(self.model.encoder.parameters()) +
                                    list(self.model.decoder.parameters()), lr=vae_lr, weight_decay=weight_decay)
        self.dis_optim = optim.Adam(list(self.model.domain_classifier.parameters()),
                                    lr=dis_lr, weight_decay=weight_decay)

        self.c_domain = c_domain
        self.c_cycle = c_cycle
        self.c_latent = c_latent
        self.c_paired = c_paired
        self.c_src_forward = c_src_forward
        self.c_trg_forward = c_trg_forward
        self.c_src_inverse = c_src_inverse
        self.c_trg_inverse = c_trg_inverse
        self.beta = beta

        self.reparam_for_losses = reparam_for_losses
        self.num_classifier_steps_per_vae_update = \
            num_classifier_steps_per_vae_update

        # training infos
        self._classifier_steps = 0
        self._classifier_metrics = []
        self._train_vae_metrics = []
        self._eval_vae_metrics = {}

    def train_epoch(self, source_loader, target_loader, paired_loader, num_train_steps):
        for _ in range(num_train_steps):
            source_batch = source_loader.sample()
            target_batch = target_loader.sample()
            paired_batch = paired_loader.sample()
            source_batch = buffer_to(source_batch, ptu.device)
            target_batch = buffer_to(target_batch, ptu.device)
            paired_batch = buffer_to(paired_batch, ptu.device)
            self.model.train_mode('discriminator')
            self._classifier_metrics.append(self._update_classifier(source_batch, target_batch))
            self.model.eval_mode('discriminator')
            self._classifier_steps += 1

            if self._classifier_steps % self.num_classifier_steps_per_vae_update == 0:
                self.model.train_mode('transfer')
                self._train_vae_metrics.append(self._update_vae(source_batch, target_batch, paired_batch))
                self.model.eval_mode('transfer')

    def eval_epoch(self, source_loader, target_loader):
        self.model.eval_mode('all')
        len_loader = min(len(source_loader), len(target_loader))
        source_loader = iter(source_loader)
        target_loader = iter(target_loader)

        metrics = []
        for i in range(len_loader):
            source_batch = next(source_loader)
            target_batch = next(target_loader)
            source_batch = buffer_to(source_batch, ptu.device)
            target_batch = buffer_to(target_batch, ptu.device)

            source_image = source_batch['image_obses'].permute(0, 3, 1, 2) / 255. - 0.5
            target_image = target_batch['image_obses'].permute(0, 3, 1, 2) / 255. - 0.5

            with torch.no_grad():
                source_output = self.model(source_image, 'source')
                target_output = self.model(target_image, 'target')

            log_prob = self.model.logprob(source_image, source_output['recon_dists']) + \
                       self.model.logprob(target_image, target_output['recon_dists'])
            kl_div = self.model.kl_divergence(source_output['latent_dists']) + \
                     self.model.kl_divergence(target_output['latent_dists'])
            latent_norm = torch.norm(source_output['latents'], p=2, dim=-1).mean() + \
                          torch.norm(target_output['latents'], p=2, dim=-1).mean()

            metrics.append({
                'eval/Log Prob': log_prob.item() / 2.0,
                'eval/KL Div': kl_div.item() / 2.0,
                'eval/Latent Norm': latent_norm.item() / 2.0
            })

        self._eval_vae_metrics = get_metric_avg(metrics)

    def sample(self, source_imgs, target_imgs):
        self.model.eval_mode('all')

        with torch.no_grad():
            source_output = self.model(source_imgs, 'source')
            target_output = self.model(target_imgs, 'target')

            target_fake = self.model.decode(source_output['latents'], 'target')
            source_fake = self.model.decode(target_output['latents'], 'source')

        sample = {
            'source_recon': source_output['recons'],
            'target_recon': target_output['recons'],
            'source_fake': source_fake[0],
            'target_fake': target_fake[0]
        }

        return sample

    def _update_vae(self, source_batch, target_batch, paired_batch):
        model = self.model
        source_image = source_batch['image_obses'].permute(0, 3, 1, 2) / 255. - 0.5
        n_source_image = source_batch['next_image_obses'].permute(0, 3, 1, 2) / 255. - 0.5
        target_image = target_batch['image_obses'].permute(0, 3, 1, 2) / 255. - 0.5
        n_target_image = target_batch['next_image_obses'].permute(0, 3, 1, 2) / 255. - 0.5

        source_paired_image = paired_batch['source_image_obses'].permute(0, 3, 1, 2) / 255. - 0.5
        target_paired_image = paired_batch['target_image_obses'].permute(0, 3, 1, 2) / 255. - 0.5

        source_action = source_batch['actions']
        target_action = target_batch['actions']

        source_output = model(source_image, 'source')
        n_source_output = model(n_source_image, 'source')
        target_output = model(target_image, 'target')
        n_target_output = model(n_target_image, 'target')

        log_prob = model.logprob(source_image, source_output['recon_dists']) + \
                   model.logprob(n_source_image, n_source_output['recon_dists']) + \
                   model.logprob(target_image, target_output['recon_dists']) + \
                   model.logprob(n_target_image, n_target_output['recon_dists'])
        kl_div = model.kl_divergence(source_output['latent_dists']) + \
                 model.kl_divergence(n_source_output['latent_dists']) + \
                 model.kl_divergence(target_output['latent_dists']) + \
                 model.kl_divergence(n_target_output['latent_dists'])
        latent_norm = torch.norm(source_output['latents'], p=2, dim=-1).mean() + \
                      torch.norm(target_output['latents'], p=2, dim=-1).mean()

        source_latent = source_output['latents'] if self.reparam_for_losses \
            else source_output['latent_dists'][0]
        n_source_latent = n_source_output['latents'] if self.reparam_for_losses \
            else n_source_output['latent_dists'][0]
        target_latent = target_output['latents'] if self.reparam_for_losses \
            else target_output['latent_dists'][0]
        n_target_latent = n_target_output['latents'] if self.reparam_for_losses \
            else n_target_output['latent_dists'][0]

        source_latent = torch.cat([source_latent, n_source_latent], dim=0)
        target_latent = torch.cat([target_latent, n_target_latent], dim=0)

        classifier_loss = model.classifier_loss(source_latent, target_latent)

        image_output = model.cycle_loss(source_image, target_image)
        n_image_output = model.cycle_loss(n_source_image, n_target_image)

        cat_source_obs = torch.stack([image_output['source_gen_image'], n_image_output['source_gen_image']], dim=0)
        cat_target_obs = torch.stack([image_output['target_gen_image'], n_image_output['target_gen_image']], dim=0)
        cat_source_act = torch.stack([torch.zeros_like(source_action), source_action], dim=0)
        cat_target_act = torch.stack([torch.zeros_like(target_action), target_action], dim=0)

        src_forward_loss = model.forward_loss(cat_source_obs, cat_target_act, 'source')
        trg_forward_loss = model.forward_loss(cat_target_obs, cat_source_act, 'target')
        src_inverse_loss = model.inverse_loss(image_output['source_gen_image'], n_image_output['source_gen_image'],
                                              target_action, 'source')
        trg_inverse_loss = model.inverse_loss(image_output['target_gen_image'], n_image_output['target_gen_image'],
                                              source_action, 'target')

        recon_loss = image_output['recon_loss'] + n_image_output['recon_loss']
        latent_loss = image_output['latent_loss'] + n_image_output['latent_loss']

        src_paired_latent = model(source_paired_image, 'source')['latents']
        trg_paired_latent = model(target_paired_image, 'target')['latents']
        paired_loss = torch.sum((src_paired_latent - trg_paired_latent) ** 2, dim=-1).mean()

        loss = -1. * log_prob / 4. + self.beta * kl_div / 4. - self.c_domain * classifier_loss / 2. + \
               self.c_cycle * recon_loss / 4. + self.c_latent * latent_loss / 4. + \
               self.c_src_forward * src_forward_loss + self.c_trg_forward * trg_forward_loss + \
               self.c_src_inverse * src_inverse_loss + self.c_trg_inverse * trg_inverse_loss + \
               self.c_paired * paired_loss

        self.vae_optim.zero_grad()
        loss.backward()
        self.vae_optim.step()

        metrics = {
            'train/Log Prob': log_prob.item() / 4.0,
            'train/KL Div': kl_div.item() / 4.0,
            'train/Domain Loss': classifier_loss.item() / 2.0,
            'train/Cycle Image Loss': recon_loss.item() / 4.0,
            'train/Cycle Latent Loss': latent_loss.item() / 4.0,
            'train/Latent Norm': latent_norm.item() / 2.0,
            'train/Src Forward Loss': src_forward_loss.item(),
            'train/Trg Forward Loss': trg_forward_loss.item(),
            'train/Src Inverse Loss': src_inverse_loss.item(),
            'train/Trg Inverse Loss': trg_inverse_loss.item(),
            'train/Paired Loss': paired_loss.item()
        }
        return metrics

    def _update_classifier(self, source_batch, target_batch):
        model = self.model
        source_image = source_batch['image_obses'].permute(0, 3, 1, 2) / 255. - 0.5
        n_source_image = source_batch['next_image_obses'].permute(0, 3, 1, 2) / 255. - 0.5
        target_image = target_batch['image_obses'].permute(0, 3, 1, 2) / 255. - 0.5
        n_target_image = target_batch['next_image_obses'].permute(0, 3, 1, 2) / 255. - 0.5

        with torch.no_grad():
            source_latent_dist = model.encode(source_image, 'source')
            n_source_latent_dist = model.encode(n_source_image, 'source')
            target_latent_dist = model.encode(target_image, 'target')
            n_target_latent_dist = model.encode(n_target_image, 'target')

            source_latent = model.reparameterize(source_latent_dist) if self.reparam_for_losses \
                else source_latent_dist[0]
            n_source_latent = model.reparameterize(n_source_latent_dist) if self.reparam_for_losses \
                else n_source_latent_dist[0]
            target_latent = model.reparameterize(target_latent_dist) if self.reparam_for_losses \
                else target_latent_dist[0]
            n_target_latent = model.reparameterize(n_target_latent_dist) if self.reparam_for_losses \
                else n_target_latent_dist[0]

        source_latent = torch.cat([source_latent, n_source_latent], dim=0)
        target_latent = torch.cat([target_latent, n_target_latent], dim=0)
        loss = model.classifier_loss(source_latent, target_latent)

        self.dis_optim.zero_grad()
        loss.backward()
        self.dis_optim.step()

        metrics = {
            'train/Classifier Loss': loss.item() / 2.
        }
        return metrics

    def get_diagnostics(self, phase='transfer'):
        metrics = {}
        metrics.update(get_metric_avg(self._classifier_metrics))
        metrics.update(get_metric_avg(self._train_vae_metrics))
        metrics.update(self._eval_vae_metrics)
        return metrics

    def end_epoch(self, epoch=0):
        self._classifier_metrics = []
        self._train_vae_metrics = []
        self._eval_vae_metrics = {}


