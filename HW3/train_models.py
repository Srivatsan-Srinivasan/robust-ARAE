# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:12:30 2018

@author: SrivatsanPC
"""
from translation_models import LSTM, LSTMA, TemporalCrossEntropyLoss
from const import *
import torch.nn as nn, torch as t
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
from collections import namedtuple
from utils import *
from torch.autograd import Variable
import json


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
    criterion = TemporalCrossEntropyLoss(size_average=False)

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
            loss = criterion(output.view(batch_size, -1, sent_length), target[:, 1:])  # @todo: do not take into account all what is after the EOS. It artificially boosts performance

            # Compute gradients, clip, and backprop
            loss.backward()
            clip_grad_norm(model.parameters(), model_params.get("clip_gradients", 5.))
            optimizer.step()

            # monitoring
            count += batch_size * sent_length  # in that case there are batch_size x bbp_length classifications per batch
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


def predict(model, test_iter, cuda=True):
    # Monitoring loss
    total_loss = 0
    count = 0
    criterion = TemporalCrossEntropyLoss(size_average=False)
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
        count += batch_size * sent_length  # in that case there are batch_size x sent_length classifications per batch
        total_loss += t.sum(loss.data)  # cut graph with .data

    # monitoring
    avg_loss = total_loss / count
    print("Validation loss is %.4f" % avg_loss)
    return avg_loss


# @todo: what's wrong with this code ? Can't tell
# def predict(model, test_iter, cuda=True, context_size=None, save_loss=False, expt_name=''):
#     model.eval()
#     total_loss = 0
#     count = 0
#     criterion = TemporalCrossEntropyLoss(size_average=False)
#     model.hidden = model.init_hidden()
#
#     for x, y in data_generator(test_iter, model.model_str, cuda=cuda):
#         pred, hidden = model(x)
#         pred = pred.permute(0, 2, 1)  # from `batch_size x bptt_length x |V|` to `batch_size x |V| x bptt_length`
#         model.hidden = model.hidden[0].detach(), model.hidden[1].detach()  # avoid memory overflows
#         total_loss += criterion.forward(pred, y).data  # .data to avoid memory overflows
#         count += x.size(0)*x.size(1)
#     if cuda:
#         total_loss = total_loss.cpu()
#     avg_loss = (total_loss / count).numpy()[0]
#     print("Validation loss is : %.4f" % avg_loss)
#     return avg_loss
