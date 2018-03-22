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


def _train_initialize_variables(model_str, model_params, opt_params, cuda, source_embedding, target_embedding):
    """Helper function that just initializes everything at the beginning of the train function"""
    # Params passed in as dict to model.
    model = eval(model_str)(model_params, source_embedding, target_embedding)
    model.train()  # important!

    optimizer = init_optimizer(opt_params, model)
    criterion = TemporalCrossEntropyLoss(size_average=False, ignore_index=PAD_TOKEN)

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


def train(model_str,
          train_iter,
          val_iter=None,
          source_embedding=None,
          target_embedding=None,
          early_stopping=False,
          save=False,
          save_path=None,
          model_params={},
          opt_params={},
          train_params={},
          cuda=CUDA_DEFAULT):
    # Initialize model and other variables
    model, criterion, optimizer, scheduler = _train_initialize_variables(model_str, model_params, opt_params, cuda, source_embedding, target_embedding)

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

            # Get data
            source = batch.src.transpose(0, 1)  # batch first
            target = batch.trg.transpose(0, 1)
            if cuda:
                source = source.cuda()
                target = target.cuda()

            # Initialize hidden layer and memory
            if model.model_str == 'LSTM':  # for LSTMA it is done in the forward because the init of the dec needs the last h of the enc
                model.hidden_enc = model.init_hidden('enc', source.size(0))
                model.hidden_dec = model.init_hidden('dec', source.size(0))

            # zero gradients
            optimizer.zero_grad()
            model.zero_grad()

            # predict
            output = model(source, target)

            # Dimension matching to cut it right for loss function.
            batch_size, sent_length = target.size(0), target.size(1)-1
            loss = criterion(output.view(batch_size, -1, sent_length), target[:, 1:])  # remove the first element of target (it is the SOS token)

            # Compute gradients, clip, and backprop
            loss.backward()
            clip_grad_norm(model.parameters(), model_params.get("clip_gradients", 5.))
            optimizer.step()

            # monitoring
            count += t.sum((target.data[:, 1:] != PAD_TOKEN).long())  # in that case there are batch_size x bbp_length classifications per batch, minus the pad tokens
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
    criterion = TemporalCrossEntropyLoss(size_average=False, ignore_index=PAD_TOKEN)
    if cuda:
        criterion = criterion.cuda()

    # Initialize hidden layer and memory
    if model.model_str == "LSTM":
        model.hidden_enc = model.init_hidden('enc')
        model.hidden_dec = model.init_hidden('dec')

    # Actual training loop.
    for batch in test_iter:
        # Get data
        source = batch.src.transpose(0, 1)  # batch first
        target = batch.trg.transpose(0, 1)
        if model.model_str == 'LSTM':  # for LSTMA it is done in the forward because the decoder needs the hidden of the encoder
            model.hidden_enc = model.init_hidden('enc', source.size(0))
            model.hidden_dec = model.init_hidden('dec', source.size(0))
        if cuda:
            source = source.cuda()
            target = target.cuda()

        # predict
        output = model.forward(source, target)

        # Dimension matching to cut it right for loss function.
        batch_size, sent_length = target.size(0), target.size(1)-1
        loss = criterion(output.view(batch_size, -1, sent_length), target[:, 1:])  # @todo: do not take into account all what is after the EOS. It artificially boosts performance

        # monitoring
        count += t.sum((target.data[:, 1:] != PAD_TOKEN).long())  # in that case there are batch_size x sent_length classifications per batch, minus the #PAD_TOKENS
        total_loss += t.sum(loss.data)  # cut graph with .data

    # monitoring
    avg_loss = total_loss / count
    print("Validation loss is %.4f" % avg_loss)
    return avg_loss


# @todo: use these two functions in the train function. They should probably be modified before
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