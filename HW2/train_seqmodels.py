# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:12:30 2018

@author: SrivatsanPC
"""
from language_models import LSTM, GRU, BiGRU, BiLSTM, TemporalCrossEntropyLoss, NNLM, NNLM2
from const import *
import torch.nn as nn, torch as t
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
from collections import namedtuple
from utils import *


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


def _train_initialize_variables(model_str, embeddings, model_params, train_iter, opt_params, cuda):
    """Helper function that just initializes everything at the beginning of the train function"""
    # Params passed in as dict to model.
    model = eval(model_str)(model_params, embeddings)
    model.train()  # important!

    if model_str == 'NNLM2':
        # in that case `train_iter` is a list of numpy arrays
        Iterator = namedtuple('Iterator', ['dataset', 'batch_size'])
        train_iter_ = Iterator(dataset=train_iter, batch_size=model_params['batch_size'])
    else:
        train_iter_ = train_iter
    optimizer = init_optimizer(opt_params, model)
    criterion = TemporalCrossEntropyLoss(size_average=False) if model.model_str != 'NNLM2' else nn.CrossEntropyLoss(size_average=False)

    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    return train_iter_, model, criterion, optimizer


def train(model_str, embeddings, train_iter, val_iter=None, context_size=None, early_stopping=False, save=False, save_path=None,
          model_params={}, opt_params={}, train_params={}, cuda=CUDA_DEFAULT, reshuffle_train=False, TEXT=None):

    # Initialize model and other variables
    train_iter_, model, criterion, optimizer = _train_initialize_variables(model_str, embeddings, model_params, train_iter, opt_params, cuda)

    # First validation round before any training
    if val_iter is not None:
        model.eval()
        valid_loss = predict(model, val_iter, valid_epochs=1, context_size=context_size,
                             save_loss=False, expt_name="dummy_expt", cuda=cuda)
        model.train()

    print("All set. Actual Training begins")
    for epoch in range(train_params.get('n_ep', 30)):
        # Monitoring loss
        total_loss = 0
        count = 0

        # if using NNLM, reshuffle sentences
        if model_str == 'NNLM':
            if reshuffle_train:
                train_iter_, _, _ = rebuild_iterators(TEXT, batch_size=int(model_params['batch_size']))

        import pdb; pdb.set_trace()
        # Actual training loop. Converting to variable later.
        if model_str in recur_models:
            hidden = (model.init_hidden()).data
            
        for x_train, y_train in data_generator(train_iter_, model_str, context_size=context_size, cuda=cuda):
            # Treating each batch as separate instance otherwise Torch accumulates gradients.
            # That could be computationally expensive.
            # Refer http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#lstm-s-in-pytorch

            if model_str in recur_models:
                model.zero_grad()
                model.hidden = variable(hidden)
                #model.hidden = hidden
            else:
                optimizer.zero_grad()

            if cuda:
                x_train = x_train.cuda()
                y_train = y_train.cuda()

            if model_str in recur_models:
                output,hidden = model(x_train)                
            else:
                output = model(x_train)

            # Dimension matching to cut it right for loss function.
            if model_str in recur_models:
                batch_size, sent_length = y_train.size(0), y_train.size(1)
                loss = criterion(output.view(batch_size, -1, sent_length), y_train)
            else:
                loss = criterion(output, y_train)

#            # backprop
#            if model in recur_models:
#                loss.backward()
#            else:
            import pdb;pdb.set_trace()
            loss.backward()
            # Clip gradients to prevent exploding gradients in RNN/LSTM/GRU
            if model_str in recur_models:
                clip_grad_norm(model.parameters(), model_params.get("clip_grad_norm", 0.25))
            optimizer.step()
            if model_str in recur_models:
                hidden = hidden.data

            # monitoring
            count += x_train.size(0)
            total_loss += t.sum(loss)

        # monitoring
        avg_loss = total_loss / count
        if cuda:
            avg_loss = avg_loss.cpu()
        print("Average loss after %d epochs is %.4f" % (epoch, avg_loss.data.numpy()[0]))
        if val_iter is not None:
            model.eval()
            former_valid_loss = valid_loss * 1.
            valid_loss = predict(model, val_iter, valid_epochs=1, context_size=context_size,
                                 save_loss=False, expt_name="dummy_expt", cuda=cuda)
            if valid_loss > former_valid_loss:
                if early_stopping:
                    break
            else:
                if save:
                    assert save_path is not None
                    save_model(model, save_path)
            model.train()

    return model


def predict(model, test_iter, valid_epochs=1, context_size=None,
            save_loss=False, expt_name="dummy_expt", cuda=CUDA_DEFAULT):
    losses = {}
    for epoch in range(valid_epochs):
        total_loss = 0
        count = 0

        if model.model_str == 'NNLM2':
            # in that case `train_iter` is a list of numpy arrays
            Iterator = namedtuple('Iterator', ['dataset', 'batch_size'])
            test_iter_ = Iterator(dataset=test_iter, batch_size=100)
        else:
            test_iter_ = test_iter

        for x_test, y_test in data_generator(test_iter_, model.model_str, context_size=context_size, cuda=cuda):
            if cuda:
                x_test = x_test.long().cuda()
                y_test = y_test.long().cuda()
            if model.model_str in recur_models:
                output, hidden = model(x_test)
            else:
                output = model(x_test)
            if model.model_str in recur_models:
                output = output.permute(0, 2, 1)

            loss = TemporalCrossEntropyLoss(size_average=False).forward(output, y_test) if model.model_str != 'NNLM2' else nn.CrossEntropyLoss(size_average=False).forward(output, y_test)
            # monitoring
            total_loss += loss
            count += x_test.size(0)

        avg_loss = total_loss / count
        if cuda:
            avg_loss = avg_loss.cpu()
        avg_loss = avg_loss.data.numpy()[0]
        if save_loss:
            losses[epoch] = avg_loss
            pickle_entry(losses, "val_loss " + expt_name)
        else:
            print("Validation loss: %4f" % avg_loss)
            losses[epoch] = avg_loss
    return losses[0]
