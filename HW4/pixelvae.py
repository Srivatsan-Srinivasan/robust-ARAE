from torch.nn import Linear as fc, ReLU, Sigmoid, Dropout, BatchNorm1d as BN
from torch import nn
from utils import variable
import torch as t
import numpy as np
import torch.nn.functional as F
from sklearn.utils import shuffle
from torchvision.utils import save_image
from matplotlib import pyplot as plt
from pixelcnn import PixelCNN


relu = ReLU()
sigmoid = Sigmoid()


class PixelVAE(nn.Module):
    """A simple VAE using BN"""

    def __init__(self, params):
        super(PixelVAE, self).__init__()
        self.model_str = 'PixelVAE'

        self.latent_dim = latent_dim = params.get('latent_dim', 2)
        self.hdim = hdim = params.get('hdim', 400)
        self.batchnorm = params.get('batchnorm', True)

        # encoder
        self.fc1 = fc(784, hdim)
        if self.batchnorm:
            self.bn_1 = BN(hdim, momentum=.9)
        self.fc_mu = fc(hdim, latent_dim)  # output the mean of z
        if self.batchnorm:
            self.bn_mu = BN(latent_dim, momentum=.9)
        self.fc_logvar = fc(hdim, latent_dim)  # output the log of the variance of z
        if self.batchnorm:
            self.bn_logvar = BN(latent_dim, momentum=.9)

        # decoder
        self.pixelcnn = PixelCNN(params)

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
            eps = variable(np.random.normal(0, 1, (len(mu), self.latent_dim)))
            return mu + std * eps
        else:
            return mu

    def decode(self, x, z, **kwargs):
        return self.pixelcnn.forward(x, z, **kwargs)

    def forward(self, x, **kwargs):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(x, z, **kwargs), mu, logvar

    def generate(self, x, z, **kwargs):
        generated_pic = x*1.

        # autoregressive generation using your own outputs as inputs
        for i in range(28):
            for j in range(28):
                xx = self.pixelcnn.forward(x, z, **kwargs)
                generated_pic[:, i, j] = xx[:, i, j]  # take only the ij-th pixel
        return generated_pic
