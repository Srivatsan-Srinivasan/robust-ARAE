# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 15:28:37 2018

@author: SrivatsanPC
"""

models = ['NNLM', 'LSTM', 'BiLSTM', 'GRU', 'Trigram']
optimizers = ['SGD', 'Adam', 'AdaMax']
recur_models = ['LSTM', 'GRU', 'BiLSTM']
embeddings = ['GloVe', 'fasttext']

CUDA_DEFAULT = False

model_params_args_map = {'num_layers'    : 'lstm_nl',
                         'hidden_dim'    : 'lstm_h_dim',
                         'embedding_dim' : 'emb_size',
                         'batch_size'    : 'batch_size',
                         'dropout'       : 'dropout'                        
                        }

opt_params_args_map = { 'optimizer' : 'optimizer',
                        'lr'        : 'lr',
                      }

train_params_args_map = { 'n_ep'     : 'n_ep'
                        }

