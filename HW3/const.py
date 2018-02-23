# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 15:28:37 2018

@author: SrivatsanPC
"""

models = ['LSTM', 'LSTMA']
optimizers = ['SGD', 'Adam', 'AdaMax']
embeddings = ['GloVe', 'fasttext']

CUDA_DEFAULT = False

model_params_args_map = {'num_layers': 'lstm_nl',
                         'hidden_dim': 'lstm_h_dim',
                         'embedding_dim': 'embedding_dim',
                         'batch_size': 'batch_size',
                         'dropout': 'dropout',
                         'train_embedding': 'emb_train',
                         'clip_grad_norm': 'clip_g_n',
                         'cuda': 'cuda',
                         'vocab_size': 'vocab_size',
                         'embed_dropout': 'embed_dropout'
                         }

opt_params_args_map = {'optimizer': 'optimizer',
                       'lr': 'lr',
                       'l2_penalty': 'l2_penalty',
                       'lr_scheduler': 'lr_scheduler'
                       }

train_params_args_map = {'n_ep': 'n_ep',
                         }
