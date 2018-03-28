from torch.nn import Linear as fc, ReLU, Sigmoid, Dropout, BatchNorm1d as BN
from torch import nn
from utils import variable
import torch as t
import numpy as np
import torch.nn.functional as F
from sklearn.utils import shuffle
from utils import one_hot


# @todo: implement GAN

relu = ReLU()
sigmoid = Sigmoid()


class Generator(nn.Module):
    """Simple generator for WGAN"""
    def __init__(self, params):
        super(Generator, self).__init__()
        self.model_str = 'WGen'

        self.latent_dim = params.get('latent_dim', 2)
        self.h_dim = params.get('hidden_dim', 100)
        self.batchnorm = params.get('batchnorm', True)

        self.fc1 = fc(self.latent_dim, self.h_dim)
        self.fc2 = fc(self.h_dim, 784)
        if self.batchnorm:
            self.bn_1 = BN(self.h_dim, eps=1e-5, momentum=.9)
            self.bn_2 = BN(784, eps=1e-5, momentum=.9)

    def forward(self, z, **kwargs):
        h = self.fc1(z)
        if self.batchnorm:
            h = F.relu(self.bn_1(h))
        else:
            h = F.relu(h)

        h = self.fc2(h)
        if self.batchnorm:
            return F.sigmoid(self.bn_2(h))
        else:
            return F.sigmoid(h)


class Discriminator(nn.Module):
    """Simple discriminator for WGAN"""
    def __init__(self, params):
        super(Discriminator, self).__init__()
        self.model_str = 'WDisc'

        self.h_dim = params.get('hidden_dim', 100)
        self.batchnorm = False
        # self.batchnorm = params.get('batchnorm', True)

        self.fc1 = fc(784, self.h_dim)
        self.fc2 = fc(self.h_dim, 1)
        if self.batchnorm:
            self.bn_1 = BN(self.h_dim, eps=1e-5, momentum=.9)
            self.bn_2 = BN(1, eps=1e-5, momentum=.9)

    def forward(self, x, **kwargs):
        h = self.fc1(x)
        if self.batchnorm:
            h = F.relu(self.bn_1(h))
        else:
            h = F.relu(h)

        h = self.fc2(h)
        # WGAN so no sigmoid at the end
        if self.batchnorm:
            return self.bn_2(h)
        else:
            return h

    def clip(self, max_weight=1.):
        assert max_weight > 0
        for p in self.parameters():
            p.data.clamp_(-max_weight, max_weight)
