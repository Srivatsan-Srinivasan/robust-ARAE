import torch as t
from torch.nn import Linear as fc, BatchNorm2d as BN2d, BatchNorm1d as BN1d
import torch.nn.functional as F


class MaskedConv2d(t.nn.Conv2d):
    """This is taken from https://github.com/jzbontar/pixelcnn-pytorch/blob/master/main.py"""
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class PixelCNN(t.nn.Module):
    """Use gated activations"""
    def __init__(self, params):
        super(PixelCNN, self).__init__()
        # get params
        self.n_layers = params.get('n_layers', 2)
        self.batchnorm = params.get('batchnorm', True)
        self.latent_dim = params.get('latent_dim', 2)
        self.filters = params.get('n_filters', 10)
        self.kernel_size = params.get('kernel_size', 3)
        self.padding = params.get('padding', 1)

        for k in range(1, self.n_layers):
            setattr(self, 'conv%da'%k, MaskedConv2d(
                'A' if k == 1 else 'B',
                in_channels=1 if k==1 else self.filters,
                out_channels=self.filters,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.padding,
                dilation=1
            ))
            setattr(self, 'conv%db' % k, MaskedConv2d(
                'A' if k == 1 else 'B',
                in_channels=1 if k==1 else self.filters,
                out_channels=self.filters,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.padding,
                dilation=1
            ))

            setattr(self, 'fc%da' % k, fc(self.latent_dim, 784))
            setattr(self, 'fc%db' % k, fc(self.latent_dim, 784))
            if self.batchnorm:
                setattr(self, 'bn_conv%da'%k, BN2d(self.filters, eps=1e-5, momentum=.9))
                setattr(self, 'bn_conv%db'%k, BN2d(self.filters, eps=1e-5, momentum=.9))
                setattr(self, 'bn_fc%da'%k, BN1d(784, eps=1e-5, momentum=.9))
                setattr(self, 'bn_fc%db'%k, BN1d(784, eps=1e-5, momentum=.9))

        self.final_conv = MaskedConv2d('B', in_channels=self.filters, out_channels=1, kernel_size=self.kernel_size, padding=self.padding, dilation=1)
        if self.batchnorm:
            self.bn_final_conv = BN2d(1, eps=1e-5, momentum=.9)

    def forward(self, x, z, **kwargs):
        h_conv = x
        for k in range(1, self.n_layers+1):
            conv_a = getattr(self, 'conv%da'%k)
            conv_b = getattr(self, 'conv%db'%k)
            fc_a = getattr(self, 'fc%da'%k)
            fc_b = getattr(self, 'fc%db'%k)
            h_conv_a = conv_a(h_conv)
            h_conv_b = conv_b(h_conv)
            h_latent_a = fc_a(z)
            h_latent_b = fc_b(z)

            if self.batchnorm:
                bn_conv_a = getattr(self, 'bn_conv%da')
                bn_conv_b = getattr(self, 'bn_conv%db')
                bn_fc_a = getattr(self, 'bn_fc%da')
                bn_fc_b = getattr(self, 'bn_fc%db')
                h_conv_a = bn_conv_a(h_conv_a)
                h_conv_b = bn_conv_b(h_conv_b)
                h_latent_a = bn_fc_a(h_latent_a)
                h_latent_b = bn_fc_b(h_latent_b)

            h_latent_a = h_latent_a.view(h_latent_a.size(0), 1, 28, 28)  # to be summed on every filter dim
            h_latent_b = h_latent_b.view(h_latent_b.size(0), 1, 28, 28)

            h_conv = F.tanh(h_conv_a + h_latent_a) * F.sigmoid(h_conv_b + h_latent_b)

        result = self.final_conv(h_conv)  # is it callable ?
        if self.batchnorm:
            return F.sigmoid(self.bn_final_conv(result))
        else:
            return F.sigmoid(result)
