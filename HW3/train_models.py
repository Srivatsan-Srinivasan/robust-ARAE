# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:12:30 2018

@author: SrivatsanPC
"""
from translation_models import LSTM, LSTMA
from const import *
import torch.nn as nn, torch as t
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
from collections import namedtuple
from utils import *
from torch.autograd import Variable
import json


# @todo: modify these functions


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


def _train_initialize_variables(model_str, embeddings, model_params, train_iter, val_iter, opt_params, cuda):
    """Helper function that just initializes everything at the beginning of the train function"""
    # Params passed in as dict to model.
    model = eval(model_str)(model_params, embeddings)
    model.train()  # important!

    if model_str == 'NNLM2':
        # in that case `train_iter` is a list of numpy arrays
        Iterator = namedtuple('Iterator', ['dataset', 'batch_size'])
        train_iter_ = Iterator(dataset=train_iter, batch_size=model_params['batch_size'])
        if val_iter is not None:
            val_iter_ = Iterator(dataset=val_iter, batch_size=model_params['batch_size'])
        else:
            val_iter_ = None
    else:
        train_iter_ = train_iter
        val_iter_ = val_iter
    optimizer = init_optimizer(opt_params, model)
    criterion = TemporalCrossEntropyLoss(size_average=False) if model.model_str != 'NNLM2' else nn.CrossEntropyLoss(size_average=False)

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
        model = model.cuda()
        criterion = criterion.cuda()
    return train_iter_, val_iter_, model, criterion, optimizer, scheduler


def train(model_str, embeddings, train_iter, val_iter=None, context_size=None, early_stopping=False, save=False, save_path=None,
          model_params={}, opt_params={}, train_params={}, cuda=CUDA_DEFAULT, reshuffle_train=False, TEXT=None):
    # Initialize model and other variables
    train_iter_, val_iter_, model, criterion, optimizer, scheduler = _train_initialize_variables(model_str, embeddings, model_params, train_iter, val_iter, opt_params, cuda)

    # First validation round before any training
    if val_iter_ is not None:
        model.eval()
        print("Model initialized")
        val_loss = predict(model, val_iter_, context_size=context_size,
                           save_loss=False, expt_name="dummy_expt", cuda=cuda)
        model.train()

    if scheduler is not None:
        scheduler.step(val_loss)

    print("All set. Actual Training begins")
    for epoch in range(train_params.get('n_ep', 30)):
        # Monitoring loss
        total_loss = 0
        count = 0

        # if using NNLM, reshuffle sentences
        if model_str == 'NNLM' and reshuffle_train:
            train_iter_, _, _ = rebuild_iterators(TEXT, batch_size=int(model_params['batch_size']))

        # Initialize hidden layer and memory(for LSTM). Converting to variable later.
        if model_str in recur_models:
            model.hidden = model.init_hidden()

        # Actual training loop.     
        for x_train, y_train in data_generator(train_iter_, model_str, context_size=context_size, cuda=cuda):

            optimizer.zero_grad()

            if model_str in recur_models:
                output, model_hidden = model(x_train)
                if model.model_str == 'LSTM':
                    model.hidden = model.hidden[0].detach(), model.hidden[1].detach()  # to break the computational graph epxlictly (backprop through `bptt_steps` steps only)
                else:
                    model.hidden = model.hidden.detach()  # to break the computational graph epxlictly (backprop through `bptt_steps` steps only)
            else:
                output = model(x_train)

            # Dimension matching to cut it right for loss function.
            if model_str in recur_models:
                batch_size, sent_length = y_train.size(0), y_train.size(1)
                loss = criterion(output.view(batch_size, -1, sent_length), y_train)
            else:
                loss = criterion(output, y_train)

            # backprop
            loss.backward()

            # Clip gradients to prevent exploding gradients in RNN/LSTM/GRU
            if model_str in recur_models:
                clip_grad_norm(model.parameters(), model_params.get("clip_grad_norm", 5))
            optimizer.step()

            # monitoring
            count += x_train.size(0) if model.model_str == 'NNLM2' else x_train.size(0) * x_train.size(1)  # in that case there are batch_size x bbp_length classifications per batch
            total_loss += t.sum(loss.data)  # .data to break so that you dont keep references

        # monitoring
        avg_loss = total_loss / count
        print("Average loss after %d epochs is %.4f" % (epoch, avg_loss))
        if val_iter_ is not None:
            model.eval()
            former_val_loss = val_loss * 1.
            val_loss = predict(model, val_iter_, context_size=context_size,
                               save_loss=False, expt_name="dummy_expt", cuda=cuda)
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


def predict(model, test_iter, cuda=True, context_size=None, save_loss=False, expt_name=''):
    # Monitoring loss
    total_loss = 0
    count = 0
    criterion = TemporalCrossEntropyLoss(size_average=False) if model.model_str != 'NNLM2' else nn.CrossEntropyLoss(size_average=False)
    if cuda:
        criterion = criterion.cuda()

    # Initialize hidden layer and memory(for LSTM). Converting to variable later.
    if model.model_str in recur_models:
        if model.model_str == "LSTM":
            h = model.init_hidden()
            hidden_init = h[0].data
            memory_init = h[1].data
        else:
            hidden_init = model.init_hidden().data

    # Actual training loop.
    for x_test, y_test in data_generator(test_iter, model.model_str, context_size=context_size, cuda=cuda):
        # Treating each batch as separate instance otherwise Torch accumulates gradients.
        # That could be computationally expensive.
        # Refer http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#lstm-s-in-pytorch
        if model.model_str in recur_models:
            # Retain hidden/memory from last batch.
            if model.model_str == 'LSTM':
                model.hidden = (variable(hidden_init, cuda=cuda), variable(memory_init, cuda=cuda))
            else:
                model.hidden = variable(hidden_init, cuda=cuda)

        if model.model_str in recur_models:
            output, model_hidden = model(x_test)
        else:
            output = model(x_test)

        # Dimension matching to cut it right for loss function.
        if model.model_str in recur_models:
            batch_size, sent_length = y_test.size(0), y_test.size(1)
            loss = criterion(output.view(batch_size, -1, sent_length), y_test)
        else:
            loss = criterion(output, y_test)

        # Remember hidden and memory for next batch. Converting to tensor to break the
        # computation graph. Converting it to variable in the next loop.
        if model.model_str in recur_models:
            if model.model_str == 'LSTM':
                hidden_init = model_hidden[0].data
                memory_init = model_hidden[1].data
            else:
                hidden_init = model_hidden.data

        # monitoring
        count += x_test.size(0) if model.model_str == 'NNLM2' else x_test.size(0) * x_test.size(1)  # in that case there are batch_size x bbp_length classifications per batch
        total_loss += t.sum(loss.data)

    # monitoring
    avg_loss = total_loss / count
    print("Validation loss is %.4f" % avg_loss)
    return avg_loss

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
