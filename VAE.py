"""VAE model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, latent_dim, device):
        """Initialize a VAE.

        Args:
            latent_dim: dimension of embedding
            device: run on cpu or gpu
        """
        super(Model, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 1, 2),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 14, 14
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  7, 7
            nn.Flatten(1),
        )

        self.mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.logvar = nn.Linear(64 * 7 * 7, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 7 * 7),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  64,  14,  14
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1, 1),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 1, 2),  # B, 1, 28, 28
            nn.Sigmoid()
        )

    def sample(self, sample_size, mu=None, logvar=None):
        '''
        :param sample_size: Number of samples
        :param mu: z mean, None for prior (init with zeros)
        :param logvar: z logstd, None for prior (init with zeros)
        :return:
        '''
        if mu == None:
            mu = torch.zeros((sample_size, self.latent_dim)).to(self.device)
        if logvar == None:
            logvar = torch.zeros((sample_size, self.latent_dim)).to(self.device)
        if mu is not None and logvar is not None:
            z = self.z_sample(mu, logvar)
        else:
            z = torch.randn(sample_size, self.latent_dim)
        return self.decoder(z)

    def z_sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        sample = torch.randn_like(std)
        z = sample * std + mu
        return z

    def loss(self, x, recon, mu, logvar):
        recons_loss = nn.functional.binary_cross_entropy(recon, x, reduction='sum')
        kl_loss = torch.sum(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
        loss = recons_loss + kl_loss
        return loss

    def forward(self, x):
        x = x.to(self.device)
        encoded = self.encoder(x)
        mu, logvar = self.mu(encoded), self.logvar(encoded)
        z = self.z_sample(mu, logvar)
        recon = self.decoder(z)
        loss = self.loss(x, recon, mu, logvar)
        return loss
