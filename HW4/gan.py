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


class GAN(nn.Module):
    pass


def loss_function(x_dec, x, mu, logvar):
    pass


def train(model, train_dataset, train_labels, epoch, batch_size, optimizer, log=100):
    pass


def test(model, test_dataset, test_labels, epoch, batch_size):
    pass
