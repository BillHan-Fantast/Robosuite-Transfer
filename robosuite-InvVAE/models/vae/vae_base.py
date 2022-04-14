import torch
import numpy as np
import torch.nn as nn
from rlkit.torch import pytorch_util as ptu


class GaussianLatentVAE(nn.Module):
    def __init__(
            self,
            representation_size,
    ):
        super().__init__()
        self.representation_size = representation_size
        self.dist_mu = np.zeros(self.representation_size)
        self.dist_std = np.ones(self.representation_size)

    def rsample(self, latent_distribution_params):
        mu, logvar = latent_distribution_params
        stds = (0.5 * logvar).exp()
        epsilon = ptu.randn(*mu.size())
        latents = epsilon * stds + mu
        return latents

    def reparameterize(self, latent_distribution_params):
        if self.training:
            return self.rsample(latent_distribution_params)
        else:
            return latent_distribution_params[0]

    def kl_divergence(self, latent_distribution_params):
        mu, logvar = latent_distribution_params
        return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

    def get_encoding_from_latent_distribution_params(self, latent_distribution_params):
        return latent_distribution_params[0].cpu()
