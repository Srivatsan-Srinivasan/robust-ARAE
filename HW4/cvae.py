from torch.nn import Linear as fc, ReLU, Sigmoid, Dropout, BatchNorm1d as BN
from torch import nn
from utils import flatten, variable
import torch as t
import numpy as np
import torch.nn.functional as F
from sklearn.utils import shuffle
from utils import one_hot
from matplotlib import pyplot as plt


# @todo: test that the main function works with this one

relu = ReLU()
sigmoid = Sigmoid()


class CVAE(nn.Module):
    """
    The only difference is that now the encoder and the decoder have an additional input
    This input consists in an array of length 10, concatenated to the flattened picture in the encoder, and to the
    latent code in the decoder
    """

    def __init__(self, latent_dim=2, hdim=400):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.hdim = hdim

        # encoder
        self.fc1 = fc(784 + 10, hdim)
        self.bn_1 = BN(hdim, momentum=.9)
        self.fc_mu = fc(hdim, latent_dim)  # output the mean of z
        self.bn_mu = BN(latent_dim, momentum=.9)
        self.fc_logvar = fc(hdim, latent_dim)  # output the variance of z
        self.bn_logvar = BN(latent_dim, momentum=.9)

        # decoder
        self.fc2 = fc(latent_dim + 10, hdim)
        self.bn_2 = BN(hdim, momentum=.9)
        self.fc3 = fc(hdim, 784)
        self.bn_3 = BN(784, momentum=.9)

    def encode(self, x, y):
        h1 = relu(self.bn_1(self.fc1(t.cat([flatten(x), y], -1))))
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

    def decode(self, z, y):
        h1 = relu(self.bn_2(self.fc2(t.cat([z, y], -1))))
        h2 = sigmoid(self.bn_3(self.fc3(h1)))
        batch_size = h2.size(0)
        x_dec = h2.resize(batch_size, 1, 28, 28)
        return x_dec

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar


def loss_function(x_dec, x, mu, logvar):
    batch_size = x_dec.size(0)
    xent = F.binary_cross_entropy(x_dec, x, size_average=True)
    kl_div = -0.5 * t.sum(1 + logvar - mu.pow(2) - t.exp(logvar))
    kl_div /= batch_size*784  # so that it is at the same scale as the xent
    return xent, kl_div


def train_one_epoch(model, train_dataset, train_labels, epoch, batch_size, optimizer, log=100):
    """
    One pass over the training dataset

    :param model:
    :param train_dataset:
    :param train_labels:
    :param epoch:
    :param batch_size:
    :param optimizer:
    :param log:
    :return:
    """
    model.train()
    train_loss = 0
    train_dataset_, train_labels_ = shuffle(train_dataset, train_labels)
    for i in range(0, len(train_dataset), batch_size):
        batch_idx = i // batch_size

        # sample data
        x = variable(t.cat(train_dataset_[i:i + batch_size], 0))
        y = variable(one_hot(t.cat(train_labels_[i:i + batch_size]).numpy()))
        if len(x) != batch_size:
            continue

        # init grads
        optimizer.zero_grad()

        x_dec, mu, logvar = model(x, y)
        xent, kl = loss_function(x_dec, x, mu, logvar)
        loss = xent + kl

        # compute grads
        loss.backward()
        train_loss += loss.data[0]

        # update weights
        optimizer.step()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss * batch_size / len(train_dataset)))
    return train_loss * batch_size / len(train_dataset_)


def test_one_epoch(model, test_dataset, test_labels, epoch, batch_size):
    """
    One pass over the test dataset
    :param model:
    :param test_dataset:
    :param test_labels:
    :param epoch:
    :param batch_size:
    :return:
    """
    # @todo: modify this function so that it saves the
    model.eval()
    test_loss = 0
    test_dataset_, test_labels_ = shuffle(test_dataset, test_labels)
    for i in range(0, len(test_dataset), batch_size):
        batch_idx = i // batch_size

        # sample data
        x = variable(t.cat(test_dataset_[i:i + batch_size], 0))
        y = variable(one_hot(t.cat(test_labels_[i:i + batch_size]).numpy()))
        if len(x) != batch_size:
            continue

        x_dec, mu, logvar = model(x, y)
        xent, kl = loss_function(x_dec, x, mu, logvar)
        test_loss += (xent + kl).data.numpy()[0]

    test_loss /= (len(test_dataset) / batch_size)
    print('====> Test set loss: {:.4f}'.format(test_loss))

    return test_loss


def generate_digit(model, n, digit):
    # @todo: modify the function so that it can save the generated images
    # generate new samples
    figure = np.zeros((28 * n, 28 * n))
    sample = variable(t.randn(n*n, model.latent_dim))
    digits = variable(one_hot(np.array(n*n*[digit])))
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
