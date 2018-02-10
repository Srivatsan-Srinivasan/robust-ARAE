# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:12:30 2018

@author: SrivatsanPC
"""
from language_models import LSTM, GRU, BiLSTM
from const import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import *


def init_optimizer(opt_params, model):
    optimizer = opt_params.get('optimizer', default='SGD')
    lr = opt_params.get('lr', default=0.1)
    if optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    if optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    if optimizer == 'Adamax':
        optimizer = optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    return optimizer


def train(model_str, embeddings, train_iter, context_size=None, model_params={}, opt_params={}, train_params={},
          cuda=CUDA_DEFAULT):
    # Params passed in as dict to model.
    model = eval(model_str)(model_params, embeddings, cuda=cuda)
    optimizer = init_optimizer(opt_params, model)
    print("All set. Actual Training begins")
    for epoch in range(train_params.get('n_ep', default=30)):
        total_loss = 0
        count = 0

        model.zero_grad()
        if model_str in recur_models:
            model.hidden = model.init_hidden()

        for x_train, y_train in data_generator(train_iter, model_str, context_size, cuda=cuda):
            # backprop
            optimizer.zero_grad()
            output = model(x_train)
            loss = F.cross_entropy(output, y_train)
            loss.backward()
            optimizer.step()

            # monitoring
            count += x_train.size(0)
            total_loss += t.sum(loss)
        avg_loss = total_loss / count
        print("Average loss after %d epochs is %.4f", (epoch, avg_loss.data.numpy()[0]))
    return model


def predict(model, model_str, test_iter, valid_epochs=10, context_size=None,
            save_loss=False, expt_name="dummy_expt", cuda=CUDA_DEFAULT):
    losses = {}
    for epoch in range(valid_epochs):
        total_loss = 0
        count = 0
        for x_test, y_test in data_generator(test_iter, model_str, context_size, cuda=cuda):
            output = model(x_test)
            loss = F.cross_entropy(output, y_test)

            # monitoring
            total_loss += loss
            count += x_test.size(0)

        avg_loss = total_loss / count
        if save_loss:
            losses[epoch] = avg_loss
            pickle_entry(losses, "val_loss" + expt_name)
        else:
            print("Avg. loss (per batch) after %d epochs is %4f", epoch, avg_loss)
    return losses
