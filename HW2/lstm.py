# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 15:46:07 2018

@author: SrivatsanPC
"""

import torch as t, numpy as np
from torch.autograd import Variable
import torch.nn as nn
from utils import *
from const import *

class LSTM():
    def __init__(self, params, embeddings, cuda = CUDA_DEFAULT) :
        super(LSTM, self).__init__(params)
        
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
        self.rnn                       = nn.LSTM(self.embedding_dim, self.hidden_dim, dropout = self.dropout)
        self.hidden2out                = nn.Linear(self.hidden_dim, self.output_size)
        self.hidden                    = self.init_hidden(self.num_layers)    
		           
    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim). The helper function
        # will return torch variable.
        return (variable(np.zeros((self.num_layers, self.batch_size, self.hidden_dim)), cuda = cuda, requires_grad = True))  
          
    def forward(self, x_batch):
        embeds = self.word_embeddings(x_batch)
        rnn_out, self.hidden = self.rnn(embeds, self.hidden_dim)
        
        #Need to train it once and check the output to match dimensions. 
        #Won't work in the present state.
        out_linear = self.hidden2out(rnn_out.view())
        
        #Use cross entropy loss on it directly.     
        return out_linear    
    
class GRU(LSTM):
    def __init__(self,params, embeddings, cuda = False ):
        LSTM.__init__(self,params)
        self.rnn = nn.GRU(self.embedding_dim, self.hidden_dim, dropout = self.dropout)
        
class BiGRU(LSTM):
     def __init__(self,params, embeddings, cuda = False ):
        LSTM.__init__(self,params)
        self.rnn = nn.GRU(self.embedding_dim, self.hidden_dim, dropout = self.dropout, bidirectional = True )
        
class BiLSTM(LSTM):
    def __init__(self,params, embeddings, cuda = False ):
        LSTM.__init__(self,params)
        self.rnn = nn.LSTM(self.embedding_dim, self.hidden_dim, dropout = self.dropout, bidirectional = True )

    
    