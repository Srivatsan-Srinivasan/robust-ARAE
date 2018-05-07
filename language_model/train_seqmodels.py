# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:12:30 2018

@author: SrivatsanPC
"""
from language_models import LSTM, TemporalCrossEntropyLoss, variable
from const import *
import torch.nn as nn, torch as t
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
import json
from utils import ReduceLROnPlateau, LambdaLR, save_model
from sklearn.utils import shuffle


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


def _train_initialize_variables(model_params, train_iter, val_iter, opt_params, ntokens, cuda, gpu_id):
    """Helper function that just initializes everything at the beginning of the train function"""
    # Params passed in as dict to model.
    model_params['ntokens'] = ntokens
    model = LSTM(model_params)
    model.train()  # important!

    train_iter_ = train_iter
    val_iter_ = val_iter
    optimizer = init_optimizer(opt_params, model)
    criterion = TemporalCrossEntropyLoss(size_average=False)

    if opt_params['lr_scheduler'] is not None and opt_params['optimizer'] == 'SGD':
        if opt_params['lr_scheduler'] == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=1)
        elif opt_params['lr_scheduler'] == 'delayedexpo':
            scheduler = LambdaLR(optimizer, lr_lambda=[lambda epoch: float(epoch<=4) + float(epoch>4)*1.2**(-epoch)])
        else:
            raise NotImplementedError('only plateau scheduler has been implemented so far')
    else:
        scheduler = None

    if cuda:
        model = model.cuda(gpu_id)
        criterion = criterion.cuda(gpu_id)
    return train_iter_, val_iter_, model, criterion, optimizer, scheduler


def train(train_iter, corpus, ntokens, val_iter=None, early_stopping=False, save=False, save_path=None, gpu_id=None,
          model_params={}, opt_params={}, train_params={}, cuda=True):
    # Initialize model and other variables
    train_iter_, val_iter_, model, criterion, optimizer, scheduler = _train_initialize_variables(model_params, train_iter, val_iter, opt_params, ntokens, cuda, gpu_id)

    # First validation round before any training
    if val_iter_ is not None:
        model.eval()
        print("Model initialized")
        val_loss = predict(model, val_iter_, cuda=cuda, gpu_id=gpu_id)
        model.train()

    if scheduler is not None:
        scheduler.step(val_loss)

    print("All set. Actual Training begins")
    for epoch in range(train_params.get('n_ep', 30)):
        # Monitoring loss
        total_loss = 0
        count = 0

        train_iter = shuffle(corpus.train)

        # Actual training loop.     
        for source, target, lengths in train_iter:
            if cuda:
                source, target = source.cuda(gpu_id), target.cuda(gpu_id)

            source, target = variable(source, cuda=cuda, gpu_id=gpu_id, to_float=False).long(), variable(target, cuda=cuda, gpu_id=gpu_id, to_float=False).long()

            optimizer.zero_grad()

            model.hidden = model.init_hidden(source.size(0))
            output, _ = model(source, lengths)
            # Dimension matching to cut it right for loss function.
            batch_size, sent_length = target.size(0), target.size(1)
            loss = criterion(output.view(batch_size, -1, sent_length), target)
            # backprop
            loss.backward()

            # Clip gradients to prevent exploding gradients in RNN/LSTM/GRU
            clip_grad_norm(model.parameters(), model_params.get("clip_grad_norm", 5))
            optimizer.step()

            # monitoring
            count += source.size(0) * source.size(1)  # in that case there are batch_size x bbp_length classifications per batch
            total_loss += t.sum(loss.data)  # .data to break so that you dont keep references

        # monitoring
        avg_loss = total_loss / count
        print("Average loss after %d epochs is %.4f" % (epoch, avg_loss))
        if val_iter_ is not None:
            model.eval()
            former_val_loss = val_loss * 1.
            val_loss = predict(model, val_iter_, cuda=cuda, gpu_id=gpu_id)
            if scheduler is not None:
                scheduler.step(val_loss)
            if val_loss > former_val_loss:
                if early_stopping:
                    break
            else:
                if save:
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


def predict(model, test_iter, cuda=True, gpu_id=None):
    # Monitoring loss
    total_loss = 0
    count = 0
    criterion = TemporalCrossEntropyLoss(size_average=False)
    if cuda:
        criterion = criterion.cuda(gpu_id)

    # Actual training loop.
    for source, target, lengths in test_iter:

        source, target = variable(source, cuda=cuda, gpu_id=gpu_id, to_float=False).long(), variable(target, cuda=cuda, gpu_id=gpu_id, to_float=False).long()
        output, _ = model(source, lengths)

        # Dimension matching to cut it right for loss function.
        print('target')
        print(target)
        print('output')
        print(output)
        batch_size, sent_length = target.size(0), target.size(1)
        loss = criterion(output.view(batch_size, -1, sent_length), target)

        # monitoring
        count += target.size(0) * target.size(1)  # in that case there are batch_size x bbp_length classifications per batch
        total_loss += t.sum(loss.data)

    # monitoring
    avg_loss = total_loss / count
    print("Validation loss is %.4f" % avg_loss)
    return avg_loss
