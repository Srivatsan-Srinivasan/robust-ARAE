from torch.nn import Linear as fc, BatchNorm1d as BN1d, BatchNorm2d as BN2d, Conv2d as conv, ConvTranspose2d as deconv, MaxPool2d as maxpool
from torch import nn
import torch.nn.functional as F
from utils import variable, flatten, freeze
import torch as t
import numpy as np
from matplotlib import pyplot as plt
from IPython import display

"""
This version is working
However note that it is subject to gradient explosion if the learning rate is too high, as there is no batch normalization
(this was shown experimentally to make the training procedure very unstable)
"""


class Generator(nn.Module):
    """Deconv generator for WGAN"""
    def __init__(self, params):
        super(Generator, self).__init__()
        self.model_str = 'WGen'

        self.latent_dim = params.get('latent_dim', 32)
        self.batchnorm = params.get('batchnorm', False)
        self.n_filters = params.get('n_filters', 32)
        self.kernel_size = params.get('kernel_size', 5)
        self.padding = params.get('padding', 2)

        self.fc1 = fc(self.latent_dim, self.n_filters*4*7*7)
        self.deconv1 = deconv(4*self.n_filters, 2*self.n_filters, self.kernel_size, stride=2, padding=self.padding, output_padding=1)
        self.deconv2 = deconv(2*self.n_filters, self.n_filters, self.kernel_size, stride=2, padding=self.padding, output_padding=1)
        self.deconv3 = deconv(self.n_filters, 1, self.kernel_size, stride=1, padding=self.padding, output_padding=0)
        if self.batchnorm:
            self.bn_fc1 = BN1d(self.n_filters*4*7*7, eps=1e-5, momentum=.9)
            self.bn_deconv1 = BN2d(2*self.n_filters, eps=1e-5, momentum=.9)
            self.bn_deconv2 = BN2d(self.n_filters, eps=1e-5, momentum=.9)
            self.bn_deconv3 = BN2d(1, eps=1e-5, momentum=.9)

    def forward(self, z, **kwargs):
        h = F.leaky_relu(self.bn_fc1(self.fc1(z))) if self.batchnorm else F.leaky_relu(self.fc1(z))
        h = h.view(h.size(0), 4*self.n_filters, 7, 7)
        h = F.leaky_relu(self.bn_deconv1(self.deconv1(h))) if self.batchnorm else F.leaky_relu(self.deconv1(h))
        h = F.leaky_relu(self.bn_deconv2(self.deconv2(h))) if self.batchnorm else F.leaky_relu(self.deconv2(h))
        h = F.sigmoid(self.bn_deconv3(self.deconv3(h))) if self.batchnorm else F.sigmoid(self.deconv3(h))
        return h


class Discriminator(nn.Module):
    """Convolutional discriminator for WGAN"""
    def __init__(self, params):
        super(Discriminator, self).__init__()
        self.model_str = 'WDisc'

        self.latent_dim = params.get('latent_dim', 32)
        self.batchnorm = params.get('batchnorm', False)
        self.n_filters = params.get('n_filters', 32)
        self.kernel_size = params.get('kernel_size', 5)
        self.padding = params.get('padding', 2)

        self.conv1 = conv(1, self.n_filters, self.kernel_size, padding=self.padding, stride=1)
        self.maxpool1 = maxpool(2, 2)
        self.conv2 = conv(self.n_filters, self.n_filters*2, self.kernel_size, padding=self.padding, stride=1)
        self.maxpool2 = maxpool(2, 2)
        self.conv3 = conv(self.n_filters*2, self.n_filters*4, self.kernel_size, padding=self.padding, stride=1)
        self.maxpool3 = maxpool(2, 2)
        self.fc1 = fc(self.n_filters*4*3*3, 1)
        if self.batchnorm:
            self.bn_conv1 = BN2d(self.n_filters, eps=1e-5, momentum=.9)
            self.bn_conv2 = BN2d(2*self.n_filters, eps=1e-5, momentum=.9)
            self.bn_conv3 = BN2d(4*self.n_filters, eps=1e-5, momentum=.9)
            self.bn_fc1 = BN1d(1, eps=1e-5, momentum=.9)

    def forward(self, x, **kwargs):
        x = x.view(x.size(0), 1, 28, 28)
        h = F.leaky_relu(self.bn_conv1(self.conv1(x))) if self.batchnorm else F.leaky_relu(self.conv1(x))
        h = self.maxpool1(h)
        h = F.leaky_relu(self.bn_conv2(self.conv2(h))) if self.batchnorm else F.leaky_relu(self.conv2(h))
        h = self.maxpool2(h)
        h = F.leaky_relu(self.bn_conv3(self.conv3(h))) if self.batchnorm else F.leaky_relu(self.conv3(h))
        h = self.maxpool3(h)
        h = h.view(h.size(0), -1)
        h = self.bn_fc1(self.fc1(h)) if self.batchnorm else self.fc1(h)
        return h.mean(0)  # PASS ONLY REAL OR ONLY FAKE SAMPLES AT A TIME

    def clip(self, max_weight=.1):
        if max_weight is None:
            return
        assert max_weight > 0
        for p in self.parameters():
            p.data.clamp_(-max_weight, max_weight)

    def _input_gradient(self, x, x_synth):
        """
        Compute gradients with regard to the input
        The input is chosen to be a random image in between the true and synthetic images
        """
        # build the input the gradients should be computed
        u = variable(np.random.uniform(size=(x.size(0), 1, 1, 1)), cuda=True)
        xx = t.autograd.Variable((x_synth * u + x * (1 - u)).data.cuda(), requires_grad=True)
        D_xx = self.forward(xx)

        # compute gradients
        gradients = t.autograd.grad(outputs=D_xx, inputs=xx,
                                    grad_outputs=t.ones(D_xx.size()).cuda(),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
        return gradients

    def gradient_penalty(self, x, x_synth, lambd=10):
        gradients = self._input_gradient(x, x_synth)
        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambd
        return gp


def monitoring(G, D, epoch, losses_g, losses_d, losses_ds, losses_dt):
    print("Epoch %d" % epoch)
    fig, axes = plt.subplots(2, 2)
    print("Losses")
    print('G|D|D Synthetic|D True')
    axes[0, 0].plot(losses_g, label='loss_g')
    axes[0, 1].plot(losses_d, label='loss_d')
    axes[1, 0].plot(losses_ds, label='loss_ds')
    axes[1, 1].plot(losses_dt, label='loss_dt')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Weights of discriminator")
    print("fc1|conv1|conv2|conv3")
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].hist(flatten(D.fc1.weight.data.cpu().numpy()), label='fc1', bins=100)
    axes[0, 1].hist(flatten(D.conv1.weight.data.cpu().numpy()), label='conv1', bins=100)
    axes[1, 0].hist(flatten(D.conv2.weight.data.cpu().numpy()), label='conv2', bins=100)
    axes[1, 1].hist(flatten(D.conv3.weight.data.cpu().numpy()), label='conv3', bins=100)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Weights of generator")
    print("fc1|deconv1|deconv2|deconv3")
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].hist(flatten(G.fc1.weight.data.cpu().numpy()), label='fc1', bins=100)
    axes[0, 1].hist(flatten(G.deconv1.weight.data.cpu().numpy()), label='deconv1', bins=100)
    axes[1, 0].hist(flatten(G.deconv2.weight.data.cpu().numpy()), label='deconv2', bins=100)
    axes[1, 1].hist(flatten(G.deconv3.weight.data.cpu().numpy()), label='deconv3', bins=100)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Gradients")
    print("Left: generator, Right: discriminator")
    print("fc1|conv1|conv2|conv3")
    fig, axes = plt.subplots(4, 2, figsize=(8, 8))
    axes[0, 1].hist(flatten(D.fc1.weight.grad.data.cpu().numpy()), label='fc1', bins=100)
    axes[1, 1].hist(flatten(D.conv1.weight.grad.data.cpu().numpy()), label='conv1', bins=100)
    axes[2, 1].hist(flatten(D.conv2.weight.grad.data.cpu().numpy()), label='conv2', bins=100)
    axes[3, 1].hist(flatten(D.conv3.weight.grad.data.cpu().numpy()), label='conv3', bins=100)
    axes[0, 0].hist(flatten(G.fc1.weight.grad.data.cpu().numpy()), label='fc1', bins=100)
    axes[1, 0].hist(flatten(G.deconv1.weight.grad.data.cpu().numpy()), label='deconv1', bins=100)
    axes[2, 0].hist(flatten(G.deconv2.weight.grad.data.cpu().numpy()), label='deconv2', bins=100)
    axes[3, 0].hist(flatten(G.deconv3.weight.grad.data.cpu().numpy()), label='deconv3', bins=100)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Generated digits")
    figure = np.zeros((28 * 5, 28 * 5))
    sample = variable(t.randn(25, G.latent_dim), cuda=True)
    G.eval()
    sample = G(sample).cpu()
    G.train()
    for k, s in enumerate(sample):
        i = k // 5
        j = k % 5
        digit = s.data.numpy().reshape(28, 28)
        figure[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = digit
    plt.imshow(figure)
    plt.show()
    display.clear_output(wait=True)


def train_wdcgan(G, D, train_loader, BATCH_SIZE):
    optimizer_d = t.optim.RMSprop(D.parameters(), lr=2e-4)
    optimizer_g = t.optim.RMSprop(G.parameters(), lr=2e-4)

    clip = None

    losses_g = []
    losses_d = []
    losses_ds = []
    losses_dt = []

    for epoch in range(250):
        for i, batch in enumerate(train_loader):
            img, label = batch
            z = variable(np.random.normal(size=(BATCH_SIZE, G.latent_dim)), cuda=True)
            img = variable(img, cuda=True)

            if i % 6 == 5:  # train gen
                # init
                D.eval()
                G.train()
                freeze(D, True)  # do not compute gradients of D
                freeze(G, False)
                optimizer_g.zero_grad()

                # forward pass
                synthetic = G(z)
                loss_s = D(synthetic)
                loss = - loss_s

                # backprop
                loss.backward()
                optimizer_g.step()

                # monitor
                losses_g.append(loss.data.cpu().numpy()[0])

                # ready to train
                D.train()

            else:  # train disc
                # init
                G.eval()
                D.train()
                freeze(D, False)
                freeze(G, True)  # do not compute gradients of G
                optimizer_d.zero_grad()

                # forward pass
                synthetic = G(z).detach()
                loss_s = D(synthetic)
                loss_t = D(img)
                loss = loss_s - loss_t + D.gradient_penalty(img, synthetic)

                # backprop
                loss.backward()
                optimizer_d.step()
                D.clip(clip)

                # monitor
                losses_d.append(loss.data.cpu().numpy()[0])
                losses_ds.append(loss_s.data.cpu().numpy()[0])
                losses_dt.append(loss_t.data.cpu().numpy()[0])

                # ready to train
                G.train()

        if epoch % 5 == 0:
            monitoring(G, D, epoch, losses_g, losses_d, losses_ds, losses_dt)
