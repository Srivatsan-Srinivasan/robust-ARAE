from torch.nn import Linear as fc, BatchNorm2d as BN2d, BatchNorm1d as BN1d, Conv2d as conv, ConvTranspose2d as deconv
from torch import nn
from utils import variable, CUDA_DEFAULT
import torch as t
import numpy as np
import torch.nn.functional as F


class ConVAE(nn.Module):
    """A convolutional VAE using BN"""

    def __init__(self, params):
        super(ConVAE, self).__init__()
        self.model_str = 'ConVAE'
        self.is_cuda = False

        self.latent_dim = latent_dim = params.get('latent_dim', 2)
        self.hdim = hdim = params.get('hdim', 400)
        self.batchnorm = params.get('batchnorm', True)
        self.kernel_size = params.get('kernel_size', 3)
        self.padding = params.get('padding', 1)
        self.n_filters = params.get('n_filters', 64)

        # encoder
        self.conv1 = conv(1, self.n_filters, self.kernel_size, padding=self.padding)
        self.conv2 = conv(self.n_filters, self.n_filters*2, self.kernel_size, padding=self.padding, stride=2)
        self.conv3 = conv(self.n_filters*2, self.n_filters*4, self.kernel_size, padding=self.padding, stride=2)
        self.fc1 = fc(7*7*self.n_filters*4, hdim)

        self.fc_mu = fc(hdim, latent_dim)  # output the mean of z
        self.fc_logvar = fc(hdim, latent_dim)  # output the log of the variance of z

        if self.batchnorm:
            self.bn_conv1 = BN2d(self.n_filters, momentum=.1)
            self.bn_conv2 = BN2d(self.n_filters*2, momentum=.1)
            self.bn_conv3 = BN2d(self.n_filters*4, momentum=.1)
            self.bn_fc1 = BN1d(self.hdim, momentum=.1)
            self.bn_mu = BN1d(latent_dim, momentum=.1)
            self.bn_logvar = BN1d(latent_dim, momentum=.1)

        # decoder
        self.fc2 = fc(latent_dim, 7*7*self.n_filters*4)
        self.deconv1 = deconv(self.n_filters*4, self.n_filters*2, self.kernel_size, stride=2, padding=self.padding, output_padding=1)
        self.deconv2 = deconv(self.n_filters*2, self.n_filters, self.kernel_size, stride=2, padding=self.padding, output_padding=1)
        self.deconv3 = deconv(self.n_filters, 1, self.kernel_size, stride=1, padding=self.padding, output_padding=0)
        if self.batchnorm:
            self.bn_fc2 = BN1d(7*7*self.n_filters*4, momentum=.1)
            self.bn_deconv1 = BN2d(self.n_filters*2, momentum=.1)
            self.bn_deconv2 = BN2d(self.n_filters, momentum=.1)
            self.bn_deconv3 = BN2d(1, momentum=.1)

    def encode(self, x, **kwargs):
        x = x.view(x.size(0), 1, 28, 28)
        h = F.leaky_relu(self.bn_conv1(self.conv1(x))) if self.batchnorm else F.leaky_relu(self.conv1(x))
        h = F.leaky_relu(self.bn_conv2(self.conv2(h))) if self.batchnorm else F.leaky_relu(self.conv2(h))
        h = F.leaky_relu(self.bn_conv3(self.conv3(h))) if self.batchnorm else F.leaky_relu(self.conv3(h))
        h = h.view(h.size(0), -1)
        h = self.fc1(h)

        mu = self.bn_mu(self.fc_mu(h)) if self.batchnorm else self.fc_mu(h)
        logvar = t.log(F.softplus(self.bn_logvar(self.fc_logvar(h)))) if self.batchnorm else t.log(F.softplus(self.fc_logvar(h)))

        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = t.exp(.5 * logvar)
            eps = variable(np.random.normal(0, 1, (len(mu), self.latent_dim)), cuda=self.is_cuda)
            return mu + std * eps
        else:
            return mu

    def decode(self, z, **kwargs):
        h = F.leaky_relu(self.bn_fc2(self.fc2(z))) if self.batchnorm else F.leaky_relu(self.fc2(z))
        h = h.view(h.size(0), self.n_filters*4, 7, 7)
        h = F.leaky_relu(self.bn_deconv1(self.deconv1(h))) if self.batchnorm else F.leaky_relu(self.deconv1(h))
        h = F.leaky_relu(self.bn_deconv2(self.deconv2(h))) if self.batchnorm else F.leaky_relu(self.deconv2(h))
        h = F.sigmoid(self.bn_deconv3(self.deconv3(h))) if self.batchnorm else F.sigmoid(self.deconv3(h))
        return h

    def forward(self, x, cuda=CUDA_DEFAULT, **kwargs):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, cuda=cuda)
        return self.decode(z), mu, logvar
