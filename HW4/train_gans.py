from wgan import Generator as WGen, Discriminator as WDisc
from const import *
import torch.nn.functional as F
import torch.optim as optim
from utils import variable, one_hot, ReduceLROnPlateau, LambdaLR, save_model
import json
import numpy as np
import os
import torch as t
from matplotlib import pyplot as plt
os.chdir('../HW4')


def init_optimizer(opt_params, model):
    optimizer = opt_params.get('optimizer', 'SGD')
    lr = opt_params.get('lr', 0.1)
    l2_penalty = opt_params.get('l2_penalty', 0)
    if optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=l2_penalty)
    if optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=l2_penalty)
    if optimizer == 'Adamax':
        optimizer = optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=l2_penalty)
    if optimizer == 'RMSProp':
        optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=l2_penalty)
    return optimizer


def _train_initialize_variables(model_str, model_params, opt_params, cuda):
    """Helper function that just initializes everything at the beginning of the train function"""
    # Params passed in as dict to model.
    if model_str == 'WGAN':
        D = WDisc(model_params)
        G = WGen(model_params)
        D.train()
        G.train()

    d_optimizer = init_optimizer(opt_params, D)
    g_optimizer = init_optimizer(opt_params, G)

    if cuda:
        D = D.cuda()
        G = G.cuda()
    return G, D, g_optimizer, d_optimizer


def _get_kwargs(model_str, train_model, z, img, label, G, D):
    """
    GAN/CGAN take different inputs for their forward method.
    As a workaround (to have a unique train function), just add `**kwargs` to each `.forward()` signature, and
    use this kwargs functions
    :returns: a dict with keys the arguments of the forward function of `model_str`
    """
    if model_str in ['CGAN', 'CWGAN']:
        if train_model == 'g':
            return {'y': label, 'z': z}
        if train_model == 'd':
            x_gen = t.bernoulli(G(z).detach())
            x = t.cat([img, x_gen], 0)
            return {'y': label, 'x': x}
    if model_str in ['GAN', 'WGAN']:
        if train_model == 'g':
            return {'z': z}
        if train_model == 'd':
            x_gen = t.bernoulli(G(z).detach())
            x = t.cat([img, x_gen], 0)
            return {'x': x}


def criterion(output, D, train_model):
    if train_model == 'd':
        batch_size = output.size(0)
        # the 1st half is the true samples, the 2nd is synthetic samples
        return - t.sum(output[:batch_size//2]) + t.sum(output[batch_size//2:])
    if train_model == 'g':
        return -t.sum(D(output))


def generate_fake_dataset(N):
    return variable(np.random.normal(size=(N, 784)))


def pretrain_disc(D, train_iter, d_optimizer, epochs=10, cuda=CUDA_DEFAULT):
    """Pretrain the discriminator"""
    for _ in range(epochs):
        for batch in train_iter:
            img, label = batch
            label = one_hot(label)
            img = img.view(img.size(0), -1)
            img, label = variable(img, cuda=cuda), variable(label, to_float=False, cuda=cuda)
            batch_size = img.size(0)

            synthetic_data = generate_fake_dataset(batch_size)
            x = t.cat([img, synthetic_data])
            labels = np.zeros((2*batch_size, )).astype(int)
            labels[:batch_size] = 1
            labels = variable(labels, cuda=cuda, to_float=False)
            output = D(x)

            loss = F.binary_cross_entropy(output, labels)

            loss.backward()
            d_optimizer.step()

            D.clip()


def train(model_str,
          train_iter,
          save=False,
          save_path=None,
          model_params={},
          opt_params={},
          train_params={},
          cuda=CUDA_DEFAULT,
          log_freq=1000
          ):

    # Initialize model and other variables
    G, D, g_optimizer, d_optimizer = _train_initialize_variables(model_str, model_params, opt_params, cuda)
    optimizers = {'g': g_optimizer, 'd': d_optimizer}
    models = {'g': G, 'd': D}
    losses = {'g': [], 'd': []}
    pretrain_disc(D, d_optimizer, train_iter, cuda=cuda)

    print("All set. Actual Training begins")
    train_model = 'd'
    max_training_iter = train_params.get('n_ep', 30) * len(train_iter)
    _ = 0
    last_log = 0
    while _ < max_training_iter:
        _ += 1

        # Monitoring loss
        total_loss = 0
        count = 0

        training_steps = 0
        # Actual training loop.
        for batch in train_iter:
            training_steps += 1  # 5 iterations for disc and 1 for gen

            # Get model
            model = models[train_model]
            optimizer = optimizers[train_model]

            # Get data
            img, label = batch
            label = one_hot(label)
            img = img.view(img.size(0), -1)
            img, label = variable(img, cuda=cuda), variable(label, to_float=False, cuda=cuda)
            batch_size = img.size(0)
            z = variable(np.random.normal(size=(batch_size, G.latent_dim)), cuda=cuda)

            if cuda:
                img = img.cuda()
                label = label.cuda()

            # zero gradients
            optimizer.zero_grad()
            model.zero_grad()

            kwargs = _get_kwargs(model_str, train_model, z, img, label, G, D)

            # Predict
            # If it is the discriminator its output has shape (2*batch size, )
            # If it is the generator its output should be passed in the discriminator to get a value.
            output = model(**kwargs)

            loss = criterion(output, D, train_model)

            # Compute gradients, clip, and backprop
            loss.backward()
            optimizer.step()
            if train_model == 'd':
                model.clip()

            # monitoring
            count += batch_size
            total_loss += t.sum(loss.data)  # .data so that you dont keep references

            if training_steps == 1 and train_model == 'g':
                losses[train_model] += [total_loss]
                train_model = 'd'
                break
            if training_steps == 5 and train_model == 'd':
                losses[train_model] += [total_loss]
                train_model = 'g'
                break

        # monitoring
        print("Average loss (past 100 values) after %d iters for %s is %.4f" % (_, 'Discr' if train_model == 'g' else 'Gen', np.mean(losses['d' if train_model == 'g' else 'g'][-100:])))
        if _ > log_freq + last_log:
            last_log = _ * 1
            G.eval()
            z = variable(np.random.normal(size=(9, G.latent_dim)), cuda=cuda)
            generated_images = G(z).data.numpy().reshape((9, 28, 28))
            np.save('%s/generated_images_%d_steps' % (save_path, _), generated_images)
            G.train()

    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    ax.plot(losses['d'], label='Disc')
    ax.plot(losses['g'], label='Gen')
    plt.legend()
    fig.savefig('%s/losses.png' % save_path)  # save the figure to file
    plt.close(fig)

    return G, D, losses
