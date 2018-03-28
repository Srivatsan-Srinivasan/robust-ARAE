models = ['GAN', 'VAE', 'CVAE', 'WGAN', 'DCGAN', 'WDCGAN', 'ConVAE']
optimizers = ['SGD', 'Adam', 'AdaMax']

CUDA_DEFAULT = False

model_params_args_map = {'num_layers': 'n_layers',
                         'hidden_dim': 'hidden_dim',
                         'latent_dim': 'embedding_dim',
                         'batch_size': 'batch_size',
                         'cuda': 'cuda',
                         'batchnorm': 'batchnorm',
                         'n_filters': 'n_filters',
                         'padding': 'padding',
                         'kernel_size': 'kernel_size'
                         }

opt_params_args_map = {'optimizer': 'optimizer',
                       'lr': 'lr',
                       'l2_penalty': 'l2_penalty',
                       'lr_scheduler': 'lr_scheduler'
                       }

train_params_args_map = {'n_ep': 'n_ep',
                         'n_pretrain': 'n_pretrain'
                         }
