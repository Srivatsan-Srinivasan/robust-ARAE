# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 15:28:37 2018

@author: SrivatsanPC
"""

models = ['NNLM', 'LSTM', 'BiLSTM', 'BiGRU', 'GRU', 'Trigram']
optimizers = ['SGD', 'Adam', 'AdaMax']
recur_models = ['LSTM', 'GRU', 'BiLSTM', 'BiGRU']
embeddings = ['GloVe', 'fasttext']

CUDA_DEFAULT = False

model_params_args_map = {'num_layers'    : 'lstm_nl',
                         'hidden_dim'    : 'lstm_h_dim',
                         'embedding_dim' : 'emb_size',
                         'batch_size'    : 'batch_size',
                         'dropout'       : 'dropout',
                         'context_size'  : 'con_size',
                         'train_embedding' : 'emb_train', 
                         'clip_grad_norm'      : 'clip_g_n',
                        }

opt_params_args_map = { 'optimizer' : 'optimizer',
                        'lr'        : 'lr',
                      }

train_params_args_map = { 'n_ep'     : 'n_ep',
                        }

