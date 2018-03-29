from vae import VAE
from cvae import CVAE
from convae import ConVAE
from pixelvae import PixelVAE
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
from IPython import display


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


def get_criterion(model_str):
    """Different models have different losses (ex: VAE=recons+KL, GAN vs WGAN...)"""
    if model_str == 'VAE' or model_str == 'CVAE' or model_str == 'PixelVAE' or model_str =='ConVAE':
        return lambda target, output: F.binary_cross_entropy_with_logits(output[0], target, size_average=False) - 0.5*t.sum(1 + output[2] - output[1].pow(2) - t.exp(output[2]))
    else:
        raise ValueError('This name is unknown: %s' % model_str)


def _train_initialize_variables(model_str, model_params, opt_params, cuda):
    """Helper function that just initializes everything at the beginning of the train function"""
    # Params passed in as dict to model.
    model = eval(model_str)(model_params)
    model.train()  # important!

    optimizer = init_optimizer(opt_params, model)
    criterion = get_criterion(model_str)

    if opt_params['lr_scheduler'] is not None:
        if opt_params['lr_scheduler'] == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=1, threshold=1e-3)
        elif opt_params['lr_scheduler'] == 'delayedexpo':
            scheduler = LambdaLR(optimizer, lr_lambda=[lambda epoch: float(epoch<=4) + float(epoch>4)*1.2**(-epoch)])
        else:
            raise NotImplementedError('only plateau scheduler has been implemented so far')
    else:
        scheduler = None

    if cuda:
        model = model.cuda()
        if 'VAE' in model_str:
            model.is_cuda = True
    return model, criterion, optimizer, scheduler


def _get_kwargs(model_str, img, label):
    """
    VAE/CVAE take different inputs for their forward method.
    As a workaround (to have a unique train function), just add `**kwargs` to each `.forward()` signature, and
    use this kwargs functions
    :returns: a dict with keys the arguments of the forward function of `model_str`
    """
    if model_str == 'CVAE':
        return {'x': img, 'y': label}
    elif model_str == 'VAE':
        return {'x': img}
    elif model_str == 'PixelVAE':
        return {'x': img}
    elif model_str == 'ConVAE':
        return {'x': img}
    else:
        raise ValueError('This name is unknown: %s' % model_str)


def train(model_str,
          train_iter,
          val_iter=None,
          early_stopping=False,
          save=False,
          save_path=None,
          model_params={},
          opt_params={},
          train_params={},
          cuda=CUDA_DEFAULT):
    # Initialize model and other variables
    model, criterion, optimizer, scheduler = _train_initialize_variables(model_str, model_params, opt_params, cuda)

    val_loss = 1e6
    best_val_loss = 1e6
    if scheduler is not None:
        assert val_iter is not None
        scheduler.step(val_loss)

    print("All set. Actual Training begins")
    for epoch in range(train_params.get('n_ep', 30)):
        # Monitoring loss
        total_loss = 0
        count = 0

        # Actual training loop.
        for batch in train_iter:
            # Get data
            img, label = batch
            label = one_hot(label)
            img = img.view(img.size(0), -1)
            img, label = variable(img, cuda=cuda), variable(label, to_float=False, cuda=cuda)
            batch_size = img.size(0)
            if cuda:
                img = img.cuda()
                label = label.cuda()

            # zero gradients
            optimizer.zero_grad()
            model.zero_grad()

            kwargs = _get_kwargs(model_str, img, label)

            # predict
            output = model(**kwargs)
            output = (output[0].view(batch_size, -1), output[1], output[2])
            loss = criterion(img, output)

            # Compute gradients, clip, and backprop
            loss.backward()
            optimizer.step()

            # monitoring
            count += batch_size
            total_loss += t.sum(loss.data)  # .data so that you dont keep references

        # monitoring
        avg_loss = total_loss / count

        model.eval()
        z = variable(np.random.normal(size=(25, model.latent_dim)), cuda=cuda)
        if 'Pixel' in model_str:
            generated_images = model.decode(variable(np.zeros((25, 1, 28, 28)), cuda=False), z).data.cpu().numpy().reshape((25, 28, 28))
        else:
            generated_images = model.decode(z).data.cpu().numpy().reshape((25, 28, 28))
        np.save('%s/generated_images_%d_steps' % (save_path, epoch), generated_images)
        model.train()

        print("Average loss after %d epochs is %.4f" % (epoch, avg_loss))
        if val_iter is not None:
            model.eval()
            former_val_loss = val_loss * 1.
            val_loss = predict(model, val_iter, cuda=cuda)
            if scheduler is not None:
                scheduler.step(val_loss)
            if val_loss > former_val_loss:
                if early_stopping:
                    break
            else:
                if save and best_val_loss > val_loss:  # save only the best
                    best_val_loss = val_loss * 1.
                    assert save_path is not None
                    # weights
                    save_model(model, save_path + '.pytorch')
                    # params
                    with open(save_path + '.params.json', 'w') as fp:
                        json.dump(model.params, fp)
                    # loss
                    with open(save_path + '.losses.txt', 'w') as fp:
                        fp.write('val: ' + str(val_loss))
                        fp.write('train: ' + str(avg_loss))
            model.train()

    return model


# @todo: generate pictures after each epoch
def predict(model, test_iter, cuda=CUDA_DEFAULT):
    # Monitoring loss
    total_loss = 0
    count = 0
    criterion = get_criterion(model.model_str)

    for batch in test_iter:
        # Get data
        img, label = batch
        label = one_hot(label)
        img = img.view(img.size(0), -1)
        img, label = variable(img, cuda=cuda), variable(label, to_float=False, cuda=cuda)
        batch_size = img.size(0)

        if cuda:
            img = img.cuda()
            label = label.cuda()

        # predict
        kwargs = _get_kwargs(model.model_str, img, label)
        output = model.forward(**kwargs)
        loss = criterion(img, output)

        # monitoring
        count += batch_size
        total_loss += t.sum(loss.data)  # cut graph with .data

    # monitoring
    avg_loss = total_loss / count
    print("Validation loss is %.4f" % avg_loss)
    return avg_loss
