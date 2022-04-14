import torch
import torch.utils.data
from torch import nn
from rlkit.pythonplusplus import identity
from rlkit.torch import pytorch_util as ptu
import numpy as np
from models.vae.conv_networks import CNN, TwoBiasDCNN, TwoHeadDCNN
from models.vae.basic_networks import DomainClassifier
from models.vae.vae_base import GaussianLatentVAE
from models.dynamics.networks import RSSMModel, InverseModel
from models.utils.tensor import infer_leading_dims
from models.combo.rnns import get_feat, get_dist


class InvVAE(GaussianLatentVAE):
    def __init__(
            self,
            image_shape,
            action_shape,
            conv_kwargs,
            deconv_kwargs,
            representation_size,
            classifier_hidden_sizes,
            decoder_output_activation=identity,
            min_variance=1e-3,
            forward_kwargs={},
            inverse_kwargs={}
    ):
        """
            customized convolution VAE class
        """
        super().__init__(representation_size)
        if min_variance is None:
            self.log_min_variance = None
        else:
            self.log_min_variance = float(np.log(min_variance))

        self.image_shape = image_shape
        self.image_length = int(np.prod(image_shape))
        self.action_shape = action_shape

        self.encoder = CNN(
            **conv_kwargs,
            output_size=2 * representation_size,
            image_shape=image_shape)

        if deconv_kwargs['architecture'] == 'TwoHead':
            self.decoder = TwoHeadDCNN(
                **deconv_kwargs,
                fc_input_size=representation_size,
                output_activation=decoder_output_activation,
            )
        elif deconv_kwargs['architecture'] == 'TwoBias':
            self.decoder = TwoBiasDCNN(
                **deconv_kwargs,
                fc_input_size=representation_size,
                output_activation=decoder_output_activation,
            )
        else:
            raise NotImplementedError

        self.domain_classifier = DomainClassifier(representation_size, classifier_hidden_sizes)
        self.forward_model = nn.ModuleDict({
            'source': RSSMModel(image_shape=image_shape, action_shape=action_shape, **forward_kwargs),
            'target': RSSMModel(image_shape=image_shape, action_shape=action_shape, **forward_kwargs)
        })

        self.inverse_model = nn.ModuleDict({
            'source': InverseModel(image_shape=image_shape, action_shape=action_shape, **inverse_kwargs),
            'target': InverseModel(image_shape=image_shape, action_shape=action_shape, **inverse_kwargs)
        })

        self._log_prob_loss = nn.MSELoss()
        self._recon_loss = nn.MSELoss()
        self._classifier_Loss = nn.NLLLoss()

    def encode(self, input, domain):
        h = self.encoder(input, domain)
        mu, logvar = h[..., :self.representation_size], \
                     h[..., self.representation_size:]
        if self.log_min_variance is not None:
            logvar = self.log_min_variance + torch.abs(logvar)
        return (mu, logvar)

    def decode(self, latents, domain):
        decoded = self.decoder(latents, domain)
        return torch.clamp(decoded, -0.5, 0.5), [torch.clamp(decoded, -0.5, 0.5), torch.ones_like(decoded)]

    def logprob(self, inputs, obs_distribution_params):
        log_prob = -1.0 * self._log_prob_loss(inputs, obs_distribution_params[0]) * self.image_length
        return log_prob

    def forward(self, input, domain):
        latent_distribution_params = self.encode(input, domain)
        latents = self.reparameterize(latent_distribution_params)
        reconstructions, obs_distribution_params = self.decode(latents, domain)

        output = {
            'recons': reconstructions,
            'latents': latents,
            'recon_dists': obs_distribution_params,
            'latent_dists': latent_distribution_params
        }
        return output

    def rollout(self, model, image_obs, action, sample=True):
        lead_dim, batch_t, batch_b, img_shape = infer_leading_dims(image_obs, 3)
        embed = model.observation_encoder(image_obs)
        prev_state = model.representation.initial_state(batch_b, device=ptu.device, dtype=action.dtype)
        prior, post = model.rollout.rollout_representation(batch_t, embed, action, prev_state, sample=sample)
        return prior, post

    def classifier_loss(self, source_input, target_input):
        source_domain_label = torch.zeros(source_input.shape[0]).long().to(ptu.device)
        target_domain_label = torch.ones(target_input.shape[0]).long().to(ptu.device)

        source_pred = self.domain_classifier(source_input)
        target_pred = self.domain_classifier(target_input)

        loss = self._classifier_Loss(source_pred, source_domain_label) \
               + self._classifier_Loss(target_pred, target_domain_label)

        return loss

    def cycle_loss(self, source_image, target_image):
        source_img_latent = self.encode(source_image, 'source')[0]
        target_img_latent = self.encode(target_image, 'target')[0]

        source_fake_recon = self.decode(target_img_latent, 'source')[0]
        target_fake_recon = self.decode(source_img_latent, 'target')[0]

        source_fake_latent = self.encode(source_fake_recon, 'source')[0]
        target_fake_latent = self.encode(target_fake_recon, 'target')[0]

        source_recon_recon = self.decode(target_fake_latent, 'source')[0]
        target_recon_recon = self.decode(source_fake_latent, 'target')[0]

        recon_loss = self._log_prob_loss(source_image, source_recon_recon) * self.image_length + \
                     self._log_prob_loss(target_image, target_recon_recon) * self.image_length

        latent_loss = torch.sum((source_img_latent - target_fake_latent) ** 2, dim=-1).mean() + \
                      torch.sum((target_img_latent - source_fake_latent) ** 2, dim=-1).mean()

        output = {
            'recon_loss': recon_loss,
            'latent_loss': latent_loss,
            'source_gen_image': source_fake_recon,
            'target_gen_image': target_fake_recon
        }

        return output

    def forward_loss(self, obs, action, domain):
        assert obs.shape[0] == 2
        model = self.forward_model[domain]
        prior, post = self.rollout(model, obs, action, sample=False)
        feat = get_feat(prior)
        pred_obs = model.observation_decoder(feat).mean[1]
        n_obs = obs[1]
        forward_loss = self._recon_loss(pred_obs, n_obs) * self.image_length
        return forward_loss

    def inverse_loss(self, obs, n_obs, action, domain):
        model = self.inverse_model[domain]
        pred_action = model(obs, n_obs)
        inverse_loss = self._recon_loss(pred_action, action) * self.action_shape[0]
        return inverse_loss

    def train_mode(self, mode):
        if mode == 'discriminator':
            self.domain_classifier.train()
        elif mode == 'dynamics_model':
            self.inverse_model.train()
            self.forward_model.train()
        elif mode == 'transfer':
            self.encoder.train()
            self.decoder.train()
        elif mode == 'all':
            self.train()
        else:
            raise NotImplementedError

    def eval_mode(self, mode):
        if mode == 'discriminator':
            self.domain_classifier.eval()
        elif mode == 'dynamics_model':
            self.inverse_model.eval()
            self.forward_model.eval()
        elif mode == 'transfer':
            self.encoder.eval()
            self.decoder.eval()
        elif mode == 'all':
            self.eval()
        else:
            raise NotImplementedError

