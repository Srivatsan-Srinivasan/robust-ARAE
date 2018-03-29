from torch.nn import Linear as fc, BatchNorm1d as BN1d, BatchNorm2d as BN2d, Conv2d as conv, ConvTranspose2d as deconv, MaxPool2d as maxpool
from torch import nn
import torch.nn.functional as F


class Generator(nn.Module):
    """Deconv generator for WGAN"""
    def __init__(self, params):
        super(Generator, self).__init__()
        self.model_str = 'WGen'

        self.latent_dim = params.get('latent_dim', 32)
        self.batchnorm = params.get('batchnorm', True)
        self.n_filters = params.get('n_filters', 32)
        self.kernel_size = params.get('kernel_size', 5)
        self.padding = params.get('padding', 2)
        self.hdim = params.get('hidden_dim', 1024)

        self.fc1 = fc(self.latent_dim, self.hdim)
        self.fc2 = fc(self.hdim, self.n_filters*2*7*7)
        self.deconv1 = deconv(2*self.n_filters, self.n_filters, self.kernel_size, stride=2, padding=self.padding, output_padding=1)
        self.deconv2 = deconv(self.n_filters, 1, self.kernel_size, stride=2, padding=self.padding, output_padding=1)
        if self.batchnorm:
            self.bn_fc1 = BN1d(self.hdim, eps=1e-5, momentum=.9)
            self.bn_fc2 = BN1d(self.n_filters*2*7*7, eps=1e-5, momentum=.9)
            self.bn_deconv1 = BN2d(self.n_filters, eps=1e-5, momentum=.9)
            self.bn_deconv2 = BN2d(1, eps=1e-5, momentum=.9)

    def forward(self, z, **kwargs):
        h = F.leaky_relu(self.bn_fc1(self.fc1(z))) if self.batchnorm else F.leaky_relu(self.fc1(z))
        h = F.leaky_relu(self.bn_fc2(self.fc2(h))) if self.batchnorm else F.leaky_relu(self.fc2(h))
        h = h.view(h.size(0), 2*self.n_filters, 7, 7)
        h = F.leaky_relu(self.bn_deconv1(self.deconv1(h))) if self.batchnorm else F.leaky_relu(self.deconv1(h))
        h = F.sigmoid(self.bn_deconv2(self.deconv2(h))) if self.batchnorm else F.sigmoid(self.deconv2(h))
        return h


class Discriminator(nn.Module):
    """Convolutional discriminator for WGAN"""
    def __init__(self, params):
        super(Discriminator, self).__init__()
        self.model_str = 'WDisc'

        self.latent_dim = params.get('latent_dim', 32)
        self.batchnorm = params.get('batchnorm', True)
        self.n_filters = params.get('n_filters', 32)
        self.hdim = params.get('hidden_dim', 1024)
        self.kernel_size = params.get('kernel_size', 5)
        self.padding = params.get('padding', 2)

        self.conv1 = conv(1, self.n_filters, self.kernel_size, padding=self.padding, stride=1)
        self.maxpool1 = maxpool(2, 2)
        self.conv2 = conv(self.n_filters, self.n_filters*2, self.kernel_size, padding=1, stride=1)
        self.maxpool2 = maxpool(2, 2)
        self.fc1 = fc(self.n_filters*2*7*7, self.hdim)
        self.fc2 = fc(self.hdim, 1)
        if self.batchnorm:
            self.bn_conv1 = BN2d(self.n_filters, eps=1e-5, momentum=.9)
            self.bn_conv2 = BN2d(2*self.n_filters, eps=1e-5, momentum=.9)
            self.bn_fc1 = BN1d(self.hdim, eps=1e-5, momentum=.9)
            self.bn_fc2 = BN1d(1, eps=1e-5, momentum=.9)

        self.make_weights_small()

    def make_weights_small(self, val=100.):
        self.conv1.weight.data = self.conv1.weight.data / val
        self.conv2.weight.data = self.conv2.weight.data / val
        self.fc1.weight.data = self.fc1.weight.data / val
        self.fc2.weight.data = self.fc2.weight.data / val

    def forward(self, x, **kwargs):
        x = x.view(x.size(0), 1, 28, 28)
        h = F.leaky_relu(self.bn_conv1(self.conv1(x))) if self.batchnorm else F.leaky_relu(self.conv1(x))
        h = self.maxpool1(h)
        h = F.leaky_relu(self.bn_conv2(self.conv2(h))) if self.batchnorm else F.leaky_relu(self.conv2(h))
        h = self.maxpool2(h)
        h = h.view(h.size(0), -1)
        h = F.leaky_relu(self.bn_fc1(self.fc1(h))) if self.batchnorm else F.leaky_relu(self.fc1(h))
        h = self.bn_fc2(self.fc2(h)) if self.batchnorm else self.fc2(h)
        return h.mean(0)  # PASS ONLY REAL OR ONLY FAKE SAMPLES AT A TIME

    def clip(self, max_weight=1e-2):
        assert max_weight > 0
        for p in self.parameters():
            p.data.clamp_(-max_weight, max_weight)
