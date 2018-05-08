models = ['BNLSTM', 'NNLM', 'NNLM2', 'LSTM', 'BiLSTM', 'BiGRU', 'GRU', 'Trigram']
optimizers = ['SGD', 'Adam', 'AdaMax']
recur_models = ['BNLSTM', 'LSTM', 'GRU', 'BiLSTM', 'BiGRU']
embeddings = ['GloVe', 'fasttext']

CUDA_DEFAULT = False

model_params_args_map = {'num_layers': 'lstm_nl',
                         'hidden_dim': 'lstm_h_dim',
                         'embedding_dim': 'emb_size',
                         'batch_size': 'batch_size',
                         'dropout': 'dropout',
                         'context_size': 'context_size',
                         'train_embedding': 'emb_train',
                         'clip_grad_norm': 'clip_g_n',
                         'cuda': 'cuda',
                         'batch_norm': 'batch_norm',
                         'nnlm_h_dim': 'nnlm_h_dim',
                         'activation': 'activation',
                         'vocab_size': 'vocab_size',
                         'embed_dropout': 'embed_dropout',
                         'tie_weights': 'tie_weights'
                         }

opt_params_args_map = {'optimizer': 'optimizer',
                       'lr': 'lr',
                       'l2_penalty': 'l2_penalty',
                       'lr_scheduler': 'lr_scheduler'
                       }

train_params_args_map = {'n_ep': 'n_ep',
                         }
