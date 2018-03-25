# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:12:30 2018

@author: SrivatsanPC
"""
from vae import VAE, test_one_epoch as test_one_epoch_vae, train_one_epoch as train_one_epoch_vae
from cvae import CVAE, test_one_epoch as test_one_epoch_cvae, train_one_epoch as train_one_epoch_cvae
from gan import GAN
from const import *
import torch.nn as nn, torch as t
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
from collections import namedtuple
from utils import *
from torch.autograd import Variable
import json
import os
from matplotlib import pyplot as plt
os.chdir('../HW4')
from IPython import display


# @todo: modify all these functions


def init_optimizer(opt_params, model):
    optimizer = opt_params.get('optimizer', 'SGD')
    lr = opt_params.get('lr', 0.1)
    l2_penalty = opt_params.get('l2_penalty', 0)
    if optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=l2_penalty)
    if optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=l2_penalty)
    if optimizer == 'Adamax':
        optimizer = optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=l2_penalty)

    return optimizer


# @todo: implement
def get_criterion(model_str, cuda):
    """Different models have different losses (ex: VAE=recons+KL, GAN vs WGAN...)"""
    pass


def _train_initialize_variables(model_str, model_params, opt_params, cuda):
    """Helper function that just initializes everything at the beginning of the train function"""
    # Params passed in as dict to model.
    model = eval(model_str)(model_params)
    model.train()  # important!

    optimizer = init_optimizer(opt_params, model)
    criterion = get_criterion(model_str, cuda)

    if opt_params['lr_scheduler'] is not None:
        if opt_params['lr_scheduler'] == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=1, threshold=1e-3)
        elif opt_params['lr_scheduler'] == 'delayedexpo':
            scheduler = LambdaLR(optimizer, lr_lambda=[lambda epoch: float(epoch<=4) + float(epoch>4)*1.2**(-epoch)])
        else:
            raise NotImplementedError('only plateau scheduler has been implemented so far')
    else:
        scheduler = None

    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    return model, criterion, optimizer, scheduler


# @todo: implement
def get_kwargs(model_str, img, label):
    """
    VAE/CVAE/GAN/CGAN/... take different inputs for their forward method.
    As a workaround (to have a unique train function), just add `**kwargs` to each `.forward()` signature, and
    use this kwargs functions
    :returns: a dict with keys the arguments of the forward function of `model_str`
    """
    pass


def train(model_str,
          train_iter,
          val_iter=None,
          early_stopping=False,
          save=False,
          save_path=None,
          model_params={},
          opt_params={},
          train_params={},
          cuda=CUDA_DEFAULT):
    # Initialize model and other variables
    model, criterion, optimizer, scheduler = _train_initialize_variables(model_str, model_params, opt_params, cuda)

    val_loss = 1e6
    best_val_loss = 1e6
    if scheduler is not None:
        assert val_iter is not None
        scheduler.step(val_loss)

    print("All set. Actual Training begins")
    for epoch in range(train_params.get('n_ep', 30)):
        # Monitoring loss
        total_loss = 0
        count = 0

        # Actual training loop.
        for batch in train_iter:
            img, label = batch
            batch_size = img.size(0)

            # Get data
            if cuda:
                img = img.cuda()
                label = label.cuda()

            # zero gradients
            optimizer.zero_grad()
            model.zero_grad()

            kwargs = get_kwargs(model_str, img, label)

            # predict
            output = model(**kwargs)  # it is a tuple for VAE

            loss = criterion(output)

            # Compute gradients, clip, and backprop
            loss.backward()
            optimizer.step()

            # monitoring
            count += batch_size
            total_loss += t.sum(loss.data)  # .data so that you dont keep references

        # monitoring
        avg_loss = total_loss / count
        print("Average loss after %d epochs is %.4f" % (epoch, avg_loss))
        if val_iter is not None:
            model.eval()
            former_val_loss = val_loss * 1.
            val_loss = predict(model, val_iter, cuda=cuda)
            if scheduler is not None:
                scheduler.step(val_loss)
            if val_loss > former_val_loss:
                if early_stopping:
                    break
            else:
                if save and best_val_loss > val_loss:  # save only the best
                    best_val_loss = val_loss * 1.
                    assert save_path is not None
                    # weights
                    save_model(model, save_path + '.pytorch')
                    # params
                    with open(save_path + '.params.json', 'w') as fp:
                        json.dump(model.params, fp)
                    # loss
                    with open(save_path + '.losses.txt', 'w') as fp:
                        fp.write('val: ' + str(val_loss))
                        fp.write('train: ' + str(avg_loss))
            model.train()

    return model


def predict(model, test_iter, cuda=True):
    # Monitoring loss
    total_loss = 0
    count = 0
    criterion = get_criterion(model.model_str, cuda)

    # Actual training loop.
    for batch in test_iter:
        # Get data
        img, label = batch
        batch_size = img.size(0)

        if cuda:
            img = img.cuda()
            label = label.cuda()

        # predict
        kwargs = get_kwargs(model.model_str, img, label)
        output = model.forward(**kwargs)

        # Dimension matching to cut it right for loss function.
        loss = criterion(output)

        # monitoring
        count += batch_size
        total_loss += t.sum(loss.data)  # cut graph with .data

    # monitoring
    avg_loss = total_loss / count
    print("Validation loss is %.4f" % avg_loss)
    return avg_loss


# @todo: use these two functions in the train function. They should probably be modified before
# HUM Maybe they are useless ?
def _train_cvae(model, optimizer, batch_size, train_dataset, train_labels, test_dataset, test_labels, EPOCHS):
    train_losses = []
    test_losses = []

    for epoch in range(1, EPOCHS + 1):
        train_losses.append(train_one_epoch_cvae(model, train_dataset, train_labels, epoch, batch_size, optimizer))
        test_losses.append(test_one_epoch_cvae(model, test_dataset, test_labels, epoch, batch_size))

        # generate new samples
        figure = np.zeros((28 * 3, 28 * 3))
        sample = variable(t.randn(9, model.latent_dim))
        digits = variable(one_hot(np.arange(1, 10, 1)))
        model.eval()
        sample = model.decode(sample, digits).cpu()
        model.train()
        for k, s in enumerate(sample):
            i = k // 3
            j = k % 3
            digit = s.data.numpy().reshape(28, 28)
            figure[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = digit

        print('epoch', epoch)
        print('losses')
        plt.plot(train_losses, label='train')
        plt.plot(test_losses, label='test')
        plt.legend()
        plt.show()
        print('generated samples')
        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap='Greys_r')
        plt.show()
        display.clear_output(wait=True)


def _train_vae(model, optimizer, batch_size, train_dataset, test_dataset, EPOCHS):
    train_losses = []
    test_losses = []

    for epoch in range(1, EPOCHS + 1):
        train_losses.append(train_one_epoch_vae(model, train_dataset, epoch, batch_size, optimizer))
        test_losses.append(test_one_epoch_vae(model, test_dataset, epoch, batch_size))

        # generate new samples
        figure = np.zeros((28 * 5, 28 * 5))
        sample = Variable(t.randn(25, model.latent_dim))
        model.eval()
        sample = model.decode(sample).cpu()
        model.train()
        for k, s in enumerate(sample):
            i = k // 5
            j = k % 5
            digit = s.data.numpy().reshape(28, 28)
            figure[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = digit

        print('epoch', epoch)
        print('losses')
        plt.plot(train_losses, label='train')
        plt.plot(test_losses, label='test')
        plt.legend()
        plt.show()
        print('generated samples')
        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap='Greys_r')
        plt.show()
        display.clear_output(wait=True)