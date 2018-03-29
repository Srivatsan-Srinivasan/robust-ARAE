from torch.nn import Linear as fc, ReLU, Sigmoid, Dropout, BatchNorm1d as BN
from torch import nn
from utils import variable
import torch as t
import numpy as np
import torch.nn.functional as F
from sklearn.utils import shuffle
from torchvision.utils import save_image
from matplotlib import pyplot as plt


relu = ReLU()
sigmoid = Sigmoid()


class VAE(nn.Module):
    """A simple VAE using BN"""

    def __init__(self, params):
        super(VAE, self).__init__()
        self.model_str = 'VAE'
        self.is_cuda = False

        self.latent_dim = latent_dim = params.get('latent_dim', 2)
        self.hdim = hdim = params.get('hdim', 400)
        self.batchnorm = params.get('batchnorm', True)

        # encoder
        self.fc1 = fc(784, hdim)
        if self.batchnorm:
            self.bn_1 = BN(hdim, momentum=.1)
        self.fc_mu = fc(hdim, latent_dim)  # output the mean of z
        if self.batchnorm:
            self.bn_mu = BN(latent_dim, momentum=.1)
        self.fc_logvar = fc(hdim, latent_dim)  # output the log of the variance of z
        if self.batchnorm:
            self.bn_logvar = BN(latent_dim, momentum=.1)

        # decoder
        self.fc2 = fc(latent_dim, hdim)
        if self.batchnorm:
            self.bn_2 = BN(hdim, momentum=.1)
        self.fc3 = fc(hdim, 784)
        if self.batchnorm:
            self.bn_3 = BN(784, momentum=.1)

    def encode(self, x, **kwargs):
        h1 = self.fc1(x)
        if self.batchnorm:
            h1 = relu(self.bn_1(h1))
        else:
            h1 = relu(h1)

        mu = self.fc_mu(h1)
        if self.batchnorm:
            mu = self.bn_mu(mu)

        logvar = self.fc_logvar(h1)
        if self.batchnorm:
            logvar = self.bn_logvar(logvar)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = t.exp(.5 * logvar)
            eps = variable(np.random.normal(0, 1, (len(mu), self.latent_dim)), cuda=self.is_cuda)
            return mu + std * eps
        else:
            return mu

    def decode(self, z, **kwargs):
        h1 = self.fc2(z)
        if self.batchnorm:
            h1 = relu(self.bn_2(h1))
        else:
            h1 = relu(h1)

        result = self.fc3(h1)
        if self.batchnorm:
            return self.bn_3(result)
        else:
            return result

    def forward(self, x, **kwargs):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
