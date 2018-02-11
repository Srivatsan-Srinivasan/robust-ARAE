# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:12:30 2018

@author: SrivatsanPC
"""
from language_models import LSTM, GRU, BiGRU, BiLSTM, TemporalCrossEntropyLoss, NNLM
from const import *
import torch.nn as nn, torch as t
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
from utils import *


def init_optimizer(opt_params, model):
    optimizer = opt_params.get('optimizer','SGD')
    lr = opt_params.get('lr', 0.1)
    if optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    if optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    if optimizer == 'Adamax':
        optimizer = optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    return optimizer


def train(model_str, embeddings, train_iter, val_iter=None, context_size = None, 
          model_params={}, opt_params={}, train_params={}, cuda=CUDA_DEFAULT):

    # Params passed in as dict to model.
    model = eval(model_str)(model_params, embeddings)
    model.train()  # important!
    optimizer = init_optimizer(opt_params, model)
    criterion = TemporalCrossEntropyLoss()
    
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
        
    print("All set. Actual Training begins")
    for epoch in range(train_params.get('n_ep', 30)):
        # monitoring variables
        total_loss = 0
        count = 0

#        model.zero_grad()
#        model.hidden = model.init_hidden() 
              
        for x_train, y_train in data_generator(train_iter, model_str, context_size = context_size, cuda=cuda):                    
            #Treating each batch as separate instance otherwise Torch accumulates gradients.
            #That could be computationally expensive.
            #Refer http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#lstm-s-in-pytorch
            if model_str in recur_models:
                model.zero_grad()
                model.hidden = model.init_hidden() 
            else:
                optimizer.zero_grad()
                
            if cuda:
                x_train = x_train.cuda()
                y_train = y_train.cuda()               
            
            output = model(x_train)        
                     
            #Dimension matching to cut it right for loss function.
            batch_size, sent_length = y_train.size()[0], y_train.size()[1]            
            if model_str in recur_models:
                loss = criterion(output.view(batch_size,-1,sent_length), y_train)
            else:
                loss = criterion(output,y_train)
            
            # backprop   
            loss.backward()
            #Clip gradients to prevent exploding gradients in RNN/LSTM/GRU
            if model_str in recur_models:
                clip_grad_norm(model.parameters(), model_params.get("clip_grad_norm", 0.25))               
            optimizer.step()           
           
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
            predict(model, val_iter, valid_epochs=1, context_size=context_size,
                    save_loss=False, expt_name="dummy_expt", cuda=cuda)
            model.train()

    return model


def predict(model, test_iter, valid_epochs=1, context_size = None,
            save_loss = False, expt_name = "dummy_expt", cuda = CUDA_DEFAULT):
    losses = {}
    for epoch in range(valid_epochs):
        total_loss = 0
        count = 0
        for x_test, y_test in data_generator(test_iter, model.model_str, context_size = context_size, cuda=cuda):
            if cuda:
                x_test = x_test.cuda()
                y_test = y_test.cuda()
            output = model(x_test)
            if model.model_str != 'NNLM':
                loss = F.cross_entropy(output, y_test)
            else:
                loss = TemporalCrossEntropyLoss().forward(output, y_test)
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
    return losses
