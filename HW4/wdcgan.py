from torch.nn import Linear as fc, ReLU, Sigmoid, Dropout, BatchNorm1d as BN1d, BatchNorm2d as BN2d, Conv2d as conv, ConvTranspose2d as deconv
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
    """Deconv generator for WGAN"""
    def __init__(self, params):
        super(Generator, self).__init__()
        self.model_str = 'WGen'

        self.latent_dim = params.get('latent_dim', 2)
        self.batchnorm = params.get('batchnorm', True)
        self.n_filters = params.get('n_filters', 20)
        self.kernel_size = params.get('kernel_size', 3)
        self.padding = params.get('padding', 1)

        self.fc1 = fc(self.latent_dim, self.n_filters*2*7*7)
        self.deconv1 = deconv(2*self.n_filters, self.n_filters, self.kernel_size, stride=2, padding=self.padding)
        self.deconv2 = deconv(self.n_filters, 1, self.kernel_size, stride=2, padding=self.padding)
        if self.batchnorm:
            self.bn_fc1 = BN1d(self.n_filters*2*7*7, eps=1e-5, momentum=.9)
            self.bn_deconv1 = BN2d(2*self.n_filters, eps=1e-5, momentum=.9)
            self.bn_deconv2 = BN2d(self.n_filters, eps=1e-5, momentum=.9)

    def forward(self, z, **kwargs):
        h = self.fc1(z)
        if self.batchnorm:
            h = F.leaky_relu(self.bn_1(h))
        else:
            h = F.leaky_relu(h)
        h = h.view(h.size(0), 2*self.n_filters, 7, 7)

        h = self.deconv1(h)
        if self.batchnorm:
            h = F.leaky_relu(self.bn_deconv1(h))
        else:
            h = F.leaky_relu(h)

        h = self.deconv2(h)
        if self.batchnorm:
            return F.leaky_relu(self.bn_deconv2(h))
        else:
            return F.leaky_relu(h)


class Discriminator(nn.Module):
    """Convolutional discriminator for WGAN"""
    def __init__(self, params):
        super(Discriminator, self).__init__()
        self.model_str = 'WDisc'

        self.latent_dim = params.get('latent_dim', 2)
        self.batchnorm = params.get('batchnorm', True)
        self.n_filters = params.get('n_filters', 20)
        self.kernel_size = params.get('kernel_size', 3)
        self.batchnorm = False
        self.padding = params.get('padding', 1)
        # self.batchnorm = params.get('batchnorm', True)

        self.conv1 = conv(1, self.n_filters, self.kernel_size, padding=1, stride=2)
        self.conv2 = conv(self.n_filters, self.n_filters*2, self.kernel_size, padding=1, stride=2)
        self.fc1 = fc(self.n_filters*2*7*7, 1)
        if self.batchnorm:
            self.bn_conv1 = BN2d(self.n_filters, eps=1e-5, momentum=.9)
            self.bn_conv2 = BN2d(2*self.n_filters, eps=1e-5, momentum=.9)
            self.bn_fc1 = BN1d(1, eps=1e-5, momentum=.9)

    def forward(self, x, **kwargs):
        h = self.conv1(x)
        if self.batchnorm:
            h = F.leaky_relu(self.bn_conv1(h))
        else:
            h = F.leaky_relu(h)

        h = self.conv2(h)
        # WGAN so no sigmoid at the end
        if self.batchnorm:
            h = F.leaky_relu(self.bn_conv2(h))
        else:
            h = F.leaky_relu(h)

        h = h.view(h.size(0), -1)
        h = self.fc1(h)
        if self.batchnorm:
            return self.bn_fc1(h)
        else:
            return h

    def clip(self, max_weight=1e-2):
        assert max_weight > 0
        for p in self.parameters():
            p.data.clamp_(-max_weight, max_weight)
