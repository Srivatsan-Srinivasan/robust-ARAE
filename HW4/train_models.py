# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:12:30 2018

@author: SrivatsanPC
"""
from vae import VAE
from cvae import CVAE
from gan import GAN
from const import *
import torch.nn.functional as F
import torch.optim as optim
from utils import variable, one_hot, ReduceLROnPlateau, LambdaLR, save_model
from torch.autograd import Variable
import json
import numpy as np
import os
import torch as t
from matplotlib import pyplot as plt
os.chdir('../HW4')
from IPython import display


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


def get_criterion(model_str):
    """Different models have different losses (ex: VAE=recons+KL, GAN vs WGAN...)"""
    if model_str == 'VAE' or model_str == 'CVAE':
        return lambda input, output: F.binary_cross_entropy(output[0], input) - 0.5*t.sum(1 + input[2] - input[1].pow(2) - t.exp(input[2]))/(784*input.size(0))


def _train_initialize_variables(model_str, model_params, opt_params, cuda):
    """Helper function that just initializes everything at the beginning of the train function"""
    # Params passed in as dict to model.
    model = eval(model_str)(model_params)
    model.train()  # important!

    optimizer = init_optimizer(opt_params, model)
    criterion = get_criterion(model_str)

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


def get_kwargs(model_str, img, label):
    """
    VAE/CVAE take different inputs for their forward method.
    As a workaround (to have a unique train function), just add `**kwargs` to each `.forward()` signature, and
    use this kwargs functions
    :returns: a dict with keys the arguments of the forward function of `model_str`
    """
    if model_str == 'CVAE':
        return {'x': img, 'y': label}
    if model_str == 'VAE':
        return {'x': img}


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
            label = one_hot(label)
            img = img.view(img.size(0), -1)
            img, label = variable(img, cuda=cuda), variable(label, to_float=False, cuda=cuda)
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
            output = model(**kwargs)
            loss = criterion(img, output)

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


def predict(model, test_iter, cuda=CUDA_DEFAULT):
    # Monitoring loss
    total_loss = 0
    count = 0
    criterion = get_criterion(model.model_str)

    for batch in test_iter:
        # Get data
        img, label = batch
        label = one_hot(label)
        img = img.view(img.size(0), -1)
        img, label = variable(img, cuda=cuda), variable(label, to_float=False, cuda=cuda)
        batch_size = img.size(0)

        if cuda:
            img = img.cuda()
            label = label.cuda()

        # predict
        kwargs = get_kwargs(model.model_str, img, label)
        output = model.forward(**kwargs)
        loss = criterion(img, output)

        # monitoring
        count += batch_size
        total_loss += t.sum(loss.data)  # cut graph with .data

    # monitoring
    avg_loss = total_loss / count
    print("Validation loss is %.4f" % avg_loss)
    return avg_loss
