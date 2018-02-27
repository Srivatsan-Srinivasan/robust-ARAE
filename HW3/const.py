# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 15:28:37 2018

@author: SrivatsanPC
"""

models = ['LSTM', 'LSTMR', 'LSTMA']
optimizers = ['SGD', 'Adam', 'AdaMax']
embeddings = ['GloVe', 'fasttext']

PAD_TOKEN = 1
SOS_TOKEN = 2
EOS_TOKEN = 3

CUDA_DEFAULT = False

model_params_args_map = {'num_layers': 'n_layers',
                         'hidden_dim': 'hidden_dim',
                         'embedding_dim': 'embedding_dim',
                         'batch_size': 'batch_size',
                         'dropout': 'dropout',
                         'train_embedding': 'emb_train',
                         'clip_gradients': 'clip_gradients',
                         'cuda': 'cuda',
                         'embed_dropout': 'embed_dropout',
                         'source_vocab_size': 'source_vocab_size',
                         'target_vocab_size': 'target_vocab_size',
                         'blstm_enc': 'blstm_enc',
                         'beam_size':'beam_size',
                         'max_beam_depth': 'max_beam_depth',
                         'embedding': 'embedding'
                         }

opt_params_args_map = {'optimizer': 'optimizer',
                       'lr': 'lr',
                       'l2_penalty': 'l2_penalty',
                       'lr_scheduler': 'lr_scheduler'
                       }

train_params_args_map = {'n_ep': 'n_ep',
                         }
