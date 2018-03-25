from torch.nn import Linear as fc, ReLU, Sigmoid, Dropout, BatchNorm1d as BN
from torch import nn
from utils import flatten, variable
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

    def __init__(self, latent_dim=2, hdim=400):
        super(VAE, self).__init__()
        self.model_str = 'VAE'
        self.latent_dim = latent_dim
        self.hdim = hdim

        # encoder
        self.fc1 = fc(784, hdim)
        self.bn_1 = BN(hdim, momentum=.9)
        self.fc_mu = fc(hdim, latent_dim)  # output the mean of z
        self.bn_mu = BN(latent_dim, momentum=.9)
        self.fc_logvar = fc(hdim, latent_dim)  # output the log of the variance of z
        self.bn_logvar = BN(latent_dim, momentum=.9)

        # decoder
        self.fc2 = fc(latent_dim, hdim)
        self.bn_2 = BN(hdim, momentum=.9)
        self.fc3 = fc(hdim, 784)
        self.bn_3 = BN(784, momentum=.9)

    def encode(self, x):
        h1 = relu(self.bn_1(self.fc1(flatten(x))))
        mu = self.bn_mu(self.fc_mu(h1))
        logvar = self.bn_logvar(self.fc_logvar(h1))
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = t.exp(.5 * logvar)
            eps = variable(np.random.normal(0, 1, (len(mu), self.latent_dim)))
            return mu + std * eps
        else:
            return mu

    def decode(self, z):
        h1 = relu(self.bn_2(self.fc2(z)))
        h2 = sigmoid(self.bn_3(self.fc3(h1)))
        batch_size = h2.size(0)
        x_dec = h2.resize(batch_size, 1, 28, 28)
        return x_dec

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(x_dec, x, mu, logvar):
    """VAE objective"""
    batch_size = x_dec.size(0)
    xent = F.binary_cross_entropy(x_dec, x, size_average=True)
    kl_div = -0.5 * t.sum(1 + logvar - mu.pow(2) - t.exp(logvar))
    kl_div /= batch_size * 784  # so that it is at the same scale as the xent
    return xent, kl_div


def train_one_epoch(model, train_dataset, epoch, batch_size, optimizer, log=100):
    """
    One pass over the training dataset
    :param model:
    :param train_dataset:
    :param epoch:
    :param batch_size:
    :param optimizer:
    :param log:
    :return:
    """
    model.train()
    train_loss = 0
    train_dataset_ = shuffle(train_dataset)
    for i in range(0, len(train_dataset), batch_size):
        batch_idx = i // batch_size

        # sample data
        x = variable(t.cat(train_dataset_[i:i + batch_size], 0))
        if len(x) != batch_size:
            continue

        # init grads
        optimizer.zero_grad()

        x_dec, mu, logvar = model(x)
        xent, kl = loss_function(x_dec, x, mu, logvar)
        loss = xent + kl

        # compute grads
        loss.backward()
        train_loss += loss.data[0]

        # update weights
        optimizer.step()
        if batch_idx % log == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_dataset),
                       100. * batch_idx / len(train_dataset),
                       loss.data[0] / len(x)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss * batch_size / len(train_dataset)))
    return train_loss * batch_size / len(train_dataset_)


def test_one_epoch(model, test_dataset, epoch, batch_size):
    """
    One pass over the test dataset
    :param model:
    :param test_dataset:
    :param epoch:
    :param batch_size:
    :return:
    """
    model.eval()
    test_loss = 0
    test_dataset_ = shuffle(test_dataset)
    for i in range(0, len(test_dataset), batch_size):
        batch_idx = i // batch_size

        # sample data
        x = variable(t.cat(test_dataset_[i:i + batch_size], 0))
        if len(x) != batch_size:
            continue

        x_dec, mu, logvar = model(x)
        xent, kl = loss_function(x_dec, x, mu, logvar)
        test_loss += (xent + kl).data.numpy()[0]
        if batch_idx == 0:
            n = min(test_dataset.size(0), 8)
            comparison = t.cat([test_dataset[:n],
                                x_dec.view(batch_size, 1, 28, 28)[:n]])
            save_image(comparison.data.cpu(),
                       'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= (len(test_dataset) / batch_size)
    print('====> Test set loss: {:.4f}'.format(test_loss))

    return test_loss


def visualize_latent(model, train_dataset, train_labels, n=10000):
    x, y = shuffle(train_dataset, train_labels)
    x = variable(t.cat(x[:n]))
    y = t.cat(y[:n]).numpy()
    mu, logvar = model.encode(x)
    z = mu.data.numpy()
    for i in range(0, 10):
        plt.scatter(z[y == i, 0], z[y == i, 1], label=str(i))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
