from torch.nn import Linear as fc, ReLU, Sigmoid, Dropout, BatchNorm1d as BN
from torch import nn
from utils import variable
import torch as t
import numpy as np
import torch.nn.functional as F
from sklearn.utils import shuffle
from utils import one_hot_np
from matplotlib import pyplot as plt


relu = ReLU()
sigmoid = Sigmoid()


class CVAE(nn.Module):
    """
    Simple CVAE
    """

    def __init__(self, params):
        super(CVAE, self).__init__()
        self.model_str = 'CVAE'

        self.latent_dim = latent_dim = params.get('latent_dim', 2)
        self.hdim = hdim = params.get('hdim', 100)
        self.batchnorm = params.get('batchnorm', True)

        # encoder
        self.fc1 = fc(784 + 10, hdim)
        if self.batchnorm:
            self.bn_1 = BN(hdim, momentum=.9)
        self.fc_mu = fc(hdim, latent_dim)  # output the mean of z
        if self.batchnorm:
            self.bn_mu = BN(latent_dim, momentum=.9)
        self.fc_logvar = fc(hdim, latent_dim)  # output the log of the variance of z
        if self.batchnorm:
            self.bn_logvar = BN(latent_dim, momentum=.9)

        # decoder
        self.fc2 = fc(latent_dim + 10, hdim)
        if self.batchnorm:
            self.bn_2 = BN(hdim, momentum=.9)
        self.fc3 = fc(hdim, 784)
        if self.batchnorm:
            self.bn_3 = BN(784, momentum=.9)

    def encode(self, x, y, **kwargs):
        h1 = self.fc1(t.cat([x, y], -1))
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

    def decode(self, z, y, **kwargs):
        h1 = self.fc2(t.cat([z, y], -1))
        if self.batchnorm:
            h1 = relu(self.bn_2(h1))
        else:
            h1 = relu(h1)

        result = self.fc3(h1)
        if self.batchnorm:
            return sigmoid(self.bn_3(result))
        else:
            return sigmoid(result)

    def forward(self, x, y, **kwargs):
        mu, logvar = self.encode(x, y, **kwargs)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y, **kwargs), mu, logvar


def loss_function(x_dec, x, mu, logvar):
    """CVAE objective"""
    batch_size = x_dec.size(0)
    xent = F.binary_cross_entropy(x_dec, x, size_average=True)
    kl_div = -0.5 * t.sum(1 + logvar - mu.pow(2) - t.exp(logvar))
    kl_div /= batch_size*784  # so that it is at the same scale as the xent
    return xent, kl_div


def generate_digit(model, n, digit):
    # @todo: modify the function so that it can save the generated images
    # generate new samples
    figure = np.zeros((28 * n, 28 * n))
    sample = variable(t.randn(n*n, model.latent_dim))
    digits = variable(one_hot_np(np.array(n*n*[digit])))
    model.eval()
    sample = model.decode(sample, digits).cpu()
    model.train()
    for k, s in enumerate(sample):
        i = k//n
        j = k%n
        digit = s.data.numpy().reshape(28, 28)
        figure[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = digit
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()
