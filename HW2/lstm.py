# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 15:46:07 2018

@author: SrivatsanPC
"""

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.vocab import Vectors, GloVe

torch.manual_seed(1)

class LSTM():
    def __init__(self, params):
        super(LSTM, self).__init__()
        
        #Initialize hyperparams.
        self.hidden_dim         = params.get('hidden_dim', default = 100)
        self.batch_size         = params.get('batch_size', default = 32)
        self.embedding_dim      = params.get('embedding_dim', default = 300)
        self.vocab_size         = params.get('vocab_size', default = 1000)
        self.output_size        = params.get('output_size' , default = self.vocab_size)
        self.num_layers         = params.get('num_layers', default = 1)
        self.dropout            = params.get('dropout', default = 0.5)
                
        #Initialize embeddings. Static embeddings for now.
        self.word_embeddings           = t.nn.Embedding(self.vocab_size, self.embedding_size)
        self.word_embeddings.weight    = nn.Parameter(embeddings, requires_grad = False)
        
        #Initialize networks.
        self.lstm                      = nn.LSTM(input_dim, hidden_dim, dropout = dropout)
        self.hidden2out                = nn.Linear(hidden_dim, output_size)
        self.hidden                    = self.init_hidden(num_layers)    
		           
    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (Variable(torch.zeros(self.num_layers,self.batch_size,self.hidden_dim)))    
          
    def forward(self, x_batch):
        embeds = self.word_embeddings(x_batch)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden_dim)
        
        #Need to train it once and check the output to match dimensions. 
        #Won't work in the present state.
        out_linear = self.hidden2out(lstm_out.view())
        
        #Use cross entropy loss on it directly.     
        return out_linear    
    
class GRU():
    

class BiLSTM():
    
    