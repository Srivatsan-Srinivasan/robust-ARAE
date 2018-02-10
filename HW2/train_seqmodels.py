# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:12:30 2018

@author: SrivatsanPC
"""
from lstm import LSTM, GRU, BiLSTM
from const import *
import torch.nn as nn
import torch.optim as optim
from utils import *

def init_optimizer(opt_params,model):
    optimizer = opt_params.get('optimizer', default = 'SGD')
    lr = opt_params.get('lr', default = 0.1)
    if optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)
    if optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)
    if optimizer == 'Adamax':
        optimizer = optim.AdaMax(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)
        
    return optimizer

def train(model_str, embeddings, train_iter, model_params = {}, opt_params = {}, train_params = {},
          cuda = CUDA_DEFAULT):
    #Params passed in as dict to model. 
    model     = eval(model_str)(model_params, embeddings, cuda = cuda)
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = init_optimizer(opt_params,model)
    print("All set. Actual Training begins")
    for epoch in range(train_params.get('n_ep', default = 30)):
        model.zero_grad()        
        if model_str in recur_models:
            model.hidden = model.init_hidden()
            
        for i, next_batch in enumerate(train_iter):
            ##UGLY PROCESSING FUNCTION FOR CONVERTING INPUTS INTO X and TARGET
            if  i == 0:
                last_batch = next_batch
                current_batch = next_batch
            x_train, y_train, last_batch, current_batch = generate_inp_out(model_str, i, next_batch, last_batch, current_batch,cuda = cuda )                       
            ##UGLY FUNCTION OVER
            
            output = model(x_train)
            loss   = loss_fn(output, y_train)
            loss.backward()
            optimizer.step()
        print("Last batch loss after %d epochs is %4f", loss)
    return model
     
def predict(model, model_str, test_iter, valid_epochs = 10, 
            save_loss=False, expt_name = "dummy_expt", cuda = CUDA_DEFAULT):
    losses = {}
    for epoch in range(valid_epochs):   
        avg_loss = 0
        for i, next_batch in enumerate(test_iter):
        ##UGLY PROCESSING FUNCTION FOR CONVERTING INPUTS INTO X and TARGET
            if  i == 0:
                last_batch = next_batch
                current_batch = next_batch
            x_test, y_test, last_batch, current_batch = generate_inp_out(model_str, i, next_batch, last_batch, current_batch, cuda = cuda )                       
        ##UGLY FUNCTION OVER
        
            output   = model(x_test)
            loss_fn  = nn.CrossEntropyLoss()
            loss     = loss_fn(output,y_test)
            avg_loss = (i*avg_loss + loss) / (i+1)

        if save_loss:
            losses[epoch] = avg_loss
            pickle_entry(losses, "val_loss"+expt_name) 
        else:
            print("Avg. loss after %d epochs is %4f", i, avg_loss)


    
    
  
    
            
        
    

