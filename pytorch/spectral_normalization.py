"""
Taken from:
https://github.com/christiancosgrove/pytorch-spectral-normalization-gan/blob/master/spectral_normalization.py
"""

import torch
from torch import nn
from torch.nn import Parameter
import numpy as np


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1, writer=None, log_freq=10000):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        self.update_count = 0
        self.log_freq = log_freq
        self.writer = writer
        if not self._made_params():
            self._make_params()

    def calc_spectral_norm(self, u, v, w, height):
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        return sigma, w / sigma.expand_as(w)

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        sigma, norm_weights = self.calc_spectral_norm(u, v, w, height)
        # norm_recomputed, weights = self.calc_spectral_norm(u, v, norm_weights, height)  # not used anyway...

        if self.update_count % self.log_freq == 0:
            true_sigma = np.linalg.norm(w.data.cpu().numpy(), 2)
            print('true_sigma.shape')
            print(true_sigma.shape)
            print('sigma.shape')
            print(sigma.data.cpu().numpy().shape)
            self.writer.add_scalar('spectral_norm_approx_'+self.name, sigma.data.cpu().numpy(), self.update_count) if self.writer is not None else None
            self.writer.add_scalar('spectral_norm_true_'+self.name, true_sigma, self.update_count) if self.writer is not None else None
        self.update_count += 1

        # Setting the weight seen by the module(in this case MLP) as spectral-normalized.
        setattr(self.module, self.name, norm_weights)

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        # Optimizer does not know of the existence of weight parameter.
        # It is deleted here in the init step and re-initialized with weight_bar.
        # Hence all gradient steps happen on weight_bar.
        del self.module._parameters[self.name]

        # Optimization(SGD) happens with respect to weight_bar
        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        # Forward gets computed using weight of the module which is defined in
        # update u_v as spectral normalized weights(check last line).
        # Optimization(SGD) happens with respect to weight_bar.
        return self.module.forward(*args)
