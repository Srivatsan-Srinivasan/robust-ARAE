import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence  # , pad_packed_sequence  # use the utis.py version instead. Useful when doing data parallelism
# from utils import pack_padded_sequence  # , pad_packed_sequence  # use the utis.py version instead. Useful when doing data parallelism
from utils import to_gpu, variable, pad_packed_sequence, Timer
import json
import os
import numpy as np
from spectral_normalization import SpectralNorm


class MLP_D(nn.Module):
    """Discriminator whose architecture is a MLP"""

    timer = Timer('Disc')

    def __init__(self, ninput, noutput, layers, activation=nn.LeakyReLU(0.2), gpu=False, weight_init='default',
                 std_minibatch=True, batchnorm=False, spectralnorm=False, writer=None, gpu_id=None, log_freq=10000,
                 lambda_GP=10, timeit=None, polar=True, dropout=None, lambda_dropout=None, sigmoid=False
                 ):
        super(MLP_D, self).__init__()
        self.ninput = ninput
        self.polar = polar
        self.noutput = noutput
        self.std_minibatch = std_minibatch
        self.gpu = gpu
        self.gpu_id = gpu_id
        self.batchnorm = batchnorm
        self.lambda_GP = lambda_GP
        self.dropout = dropout  # THIS IS ONLY FOR THE DROPOUT PENALTY FROM THE PAPER `IMPROVING THE IMPROVED TRAINING OF WGAN` !!! This is not regular dropout. Just a way of enforcing Lipschitz continuity
        self.lambda_dropout = lambda_dropout
        self.sigmoid = sigmoid
        if isinstance(activation, t.nn.ReLU):
            self.negative_slope = 0
        elif isinstance(activation, t.nn.LeakyReLU):
            self.negative_slope = activation.negative_slope
        else:
            raise ValueError('Not implemented')

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.n_layers = len(layer_sizes)

        for i in range(len(layer_sizes) - 1):
            if spectralnorm:
                layer = SpectralNorm(nn.Linear(layer_sizes[i], layer_sizes[i + 1]), writer=writer, log_freq=log_freq, logname='layer'+str(i))
            else:
                layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            setattr(self, 'layer' + str(i + 1), layer)

            # No batch normalization after first layer
            if i != 0 and batchnorm:
                bn = nn.BatchNorm1d(layer_sizes[i + 1], eps=1e-05, momentum=0.1)
                setattr(self, 'bn' + str(i + 1), bn)

            setattr(self, 'activation' + str(i + 1), activation)

        if spectralnorm:
            layer = SpectralNorm(nn.Linear(layer_sizes[-1] + int(std_minibatch), noutput), writer=writer, log_freq=log_freq, logname='layer'+str(len(layer_sizes)))
        else:
            layer = nn.Linear(layer_sizes[-1] + int(std_minibatch), noutput)

        setattr(self, 'layer' + str(self.n_layers), layer)

        self.init_weights(weight_init)

        if timeit is not None:
            MLP_D.timer.enabled = True
            MLP_D.timer.log_freq = timeit
            MLP_D.timer.writer = writer
            self.forward = MLP_D.timer.timeit(self.forward)
            self.gradient_penalty = MLP_D.timer.timeit(self.gradient_penalty)

    def forward(self, x, mean=True, writer=None):
        for i in range(1, self.n_layers):
            layer = getattr(self, 'layer%d' % i)
            activation = getattr(self, 'activation%d' % i)
            bn = getattr(self, 'bn%d' % i) if self.batchnorm and i > 1 else None
            x = activation(bn(layer(x))) if bn is not None else activation(layer(x))

        layer = getattr(self, 'layer%d' % self.n_layers)

        if self.std_minibatch:
            x_std_feature = t.mean(t.std(x, 0)).unsqueeze(1).expand(x.size(0), 1)
            x = t.cat([x, x_std_feature], 1)

        x = layer(x)
        if not self.sigmoid:
            return t.mean(x) if mean else x
        else:
            return t.mean(F.sigmoid(x)) if mean else F.sigmoid(x)

    def init_weights(self, weight_init='default'):
        if weight_init == 'default':
            init_std = 0.02
            for i in range(1, self.n_layers):
                try:
                    layer = getattr(self, 'layer' + str(i))
                    layer.weight.data.normal_(0, init_std)
                    layer.bias.data.fill_(0)
                except:
                    pass
        elif weight_init == 'he':
            for i in range(1, self.n_layers):
                try:
                    layer = getattr(self, 'layer' + str(i))
                    t.nn.init.kaiming_normal_(layer.weight.data, a=self.negative_slope)
                    layer.bias.data.fill_(0)
                except:
                    pass
        else:
            raise NotImplementedError

    def _input_gradient(self, x, x_synth):
        """
        Compute gradients with regard to the input
        The input is chosen to be a random vector in between the true and synthetic images (uniformly on the segments linking the two)

        Code taken from https://github.com/caogang/wgan-gp/blob/master/gan_mnist.py#L143
        """
        # build the input the gradients should be computed
        u = t.rand(x.size(0), 1)
        u = u.expand(x.size())
        u = u.cuda(self.gpu_id)
        x_data = x.data
        x_synth_data = x_synth.data

        if not self.polar:
            interpolate = (x_synth_data * u + x_data * (1 - u))
        else:
            interpolate = x_synth_data * t.sqrt(u) + x_data * t.sqrt(1 - u)
        interpolate = interpolate.cuda(self.gpu_id) if self.gpu else interpolate
        xx = t.autograd.Variable(interpolate, requires_grad=True)
        D_xx = self.forward(xx)

        # compute gradients
        grad_outputs = t.ones(D_xx.size())
        grad_outputs = grad_outputs.cuda(self.gpu_id) if self.gpu else grad_outputs
        gradients = t.autograd.grad(outputs=D_xx, inputs=xx,
                                    grad_outputs=grad_outputs,
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
        return gradients

    def gradient_penalty(self, x, x_synth):
        """
        Gradient penalty to force the discriminator to be 1-Lipschitz continuous

        See `Improved Training of Wasserstein GANs`: https://arxiv.org/abs/1704.00028
        Code taken from https://github.com/caogang/wgan-gp/blob/master/gan_mnist.py#L143
        """
        gradients = self._input_gradient(x, x_synth)
        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_GP
        return gp

    def dropout_penalty(self, x):
        """
        IMPROVING THE IMPROVED TRAINING OF WASSERSTEIN GANS: A CONSISTENCY TERM AND ITS DUAL EFFECT
        https://openreview.net/forum?id=SJx9GQb0-
        """
        if self.dropout is None and self.lambda_dropout is not None:
            raise ValueError('neither `self.dropout` nor `self.lambda_dropout` should not be None if you want to use '
                             'a dropout penalty to enforce Lipscitz continuity')

        x1 = x * 1.
        x2 = x * 1.

        for i in range(1, self.n_layers):
            layer = getattr(self, 'layer%d' % i)
            activation = getattr(self, 'activation%d' % i)
            bn = getattr(self, 'bn%d' % i) if self.batchnorm and i > 1 else None
            dropout = t.nn.Dropout(self.dropout)
            x1 = activation(bn(layer(dropout(x1)))) if bn is not None else activation(layer(dropout(x1)))
            dropout = t.nn.Dropout(self.dropout)
            x2 = activation(bn(layer(dropout(x2)))) if bn is not None else activation(layer(dropout(x2)))
        penultimate_x1 = x1 * 1.
        penultimate_x2 = x2 * 1.

        layer = getattr(self, 'layer%d' % self.n_layers)

        if self.std_minibatch:
            x_std_feature1 = t.mean(t.std(x1, 0)).unsqueeze(1).expand(x1.size(0), 1)
            x_std_feature2 = t.mean(t.std(x2, 0)).unsqueeze(1).expand(x2.size(0), 1)
            x1 = t.cat([x1, x_std_feature1], 1)
            x2 = t.cat([x2, x_std_feature2], 1)

        dropout = t.nn.Dropout(self.dropout)
        x1 = layer(dropout(x1))
        dropout = t.nn.Dropout(self.dropout)
        x2 = layer(dropout(x2))
        return self.lambda_dropout * ((x1 - x2).norm(2, 1) + .1 * (penultimate_x1 - penultimate_x2).norm(2, 1)).mean()

    def tensorboard(self, writer, n_iter):
        """
        Note that tensorboard default binning (for histograms) is buggy, prefer `doane` instead
        """
        for i in range(1, self.n_layers + 1):
            layer = getattr(self, 'layer%d' % i)
            if isinstance(layer, t.nn.Linear):
                writer.add_histogram('Disc_fc_w_%d' % i, layer.weight.data.cpu().numpy(), n_iter, bins='doane')
                writer.add_histogram('Disc_fc_grad_%d' % i, layer.weight.grad.cpu().data.numpy(), n_iter, bins='doane')
            elif isinstance(layer, SpectralNorm):
                writer.add_histogram('Disc_fc_w_%d' % i, layer.module.weight_bar.data.cpu().numpy(), n_iter, bins='doane')
                writer.add_histogram('Disc_fc_grad_%d' % i, layer.module.weight_bar.grad.cpu().data.numpy(), n_iter, bins='doane')
            else:
                raise ValueError('there is a problem here, it should be either Linear or SpectralNorm')

    def __cuda__(self, gpu_id=None):
        if gpu_id is None and self.gpu_id is None:
            gpu_id = 0
        if gpu_id is None and self.gpu_id is not None:
            gpu_id = self.gpu_id
        self.cuda(gpu_id)
        self.gpu_id = gpu_id
        self.gpu = True


class MLP_G(nn.Module):
    """Generator whose architecture is a MLP"""
    timer = Timer('Gen', enabled=False, log_freq=None, writer=None)

    def __init__(self, ninput, noutput, layers, activation=nn.ReLU(),
                 gpu=False, gpu_id=None, weight_init='default', batchnorm=True,
                 timeit=None, writer=None):
        super(MLP_G, self).__init__()
        self.ninput = ninput
        self.noutput = noutput
        self.batchnorm = batchnorm
        self.gpu = gpu
        self.gpu_id = gpu_id
        if isinstance(activation, t.nn.ReLU):
            self.negative_slope = 0
        elif isinstance(activation, t.nn.LeakyReLU):
            self.negative_slope = activation.negative_slope
        else:
            raise ValueError('Not implemented')

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.n_layers = len(layer_sizes)

        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            setattr(self, 'layer' + str(i + 1), layer)

            if batchnorm:
                bn = nn.BatchNorm1d(layer_sizes[i + 1], eps=1e-05, momentum=0.1)
                setattr(self, 'bn' + str(i + 1), bn)

            setattr(self, 'activation' + str(i + 1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput)
        setattr(self, 'layer' + str(self.n_layers), layer)

        self.init_weights(weight_init)

        if timeit is not None:
            MLP_G.timer.enabled = True
            MLP_G.timer.log_freq = timeit
            MLP_G.timer.writer = writer
            self.forward = MLP_G.timer.timeit(self.forward)

    def forward(self, x):
        for i in range(1, self.n_layers + 1):
            layer = getattr(self, 'layer' + str(i))
            if i == self.n_layers:
                return layer(x)
            activation = getattr(self, 'activation' + str(i))
            bn = getattr(self, 'bn' + str(i)) if self.batchnorm else None
            x = activation(bn(layer(x))) if bn is not None else activation(layer(x))
        return x

    def init_weights(self, weight_init='default'):
        if weight_init == 'default':
            init_std = 0.02
            for i in range(1, self.n_layers + 1):
                try:
                    layer = getattr(self, 'layer' + str(i))
                    layer.weight.data.normal_(0, init_std)
                    layer.bias.data.fill_(0)
                except:
                    pass
        elif weight_init == 'he':
            for i in range(1, self.n_layers + 1):
                try:
                    layer = getattr(self, 'layer' + str(i))
                    t.nn.init.kaiming_normal_(layer.weight.data, a=self.negative_slope)
                    layer.bias.data.fill_(0)
                except:
                    pass
        else:
            raise NotImplementedError('Not implemented')

    def tensorboard(self, writer, n_iter):
        """
        Note that tensorboard default binning (for histograms) is buggy, prefer `doane` instead
        """
        # layers
        for i in range(1, self.n_layers + 1):
            layer = getattr(self, 'layer%d' % i)
            if isinstance(layer, t.nn.Linear):
                writer.add_histogram('Gen_fc_w_%d' % i, layer.weight.data.cpu().numpy(), n_iter, bins='doane')
                writer.add_histogram('Gen_fc_grad_%d' % i, layer.weight.grad.cpu().data.numpy(), n_iter, bins='doane')

        # Distributional properties of the generated codes
        # l2 norm
        z = variable(np.random.normal(size=(500, self.ninput)), cuda=self.gpu, gpu_id=self.gpu_id)  # @todo: use a bigger number than 500 ?
        c = self.forward(z)
        l2norm = t.mean(t.sum(c ** 2, 1))
        writer.add_scalar('l2_norm_gen', l2norm, n_iter)
        # sum of variances
        trace_cov = np.trace(np.cov(c.data.cpu().numpy()).T)
        writer.add_scalar('trace_cov_gen', trace_cov, n_iter)

    def __cuda__(self, gpu_id=None):
        if gpu_id is None and self.gpu_id is None:
            gpu_id = 0
        if gpu_id is None and self.gpu_id is not None:
            gpu_id = self.gpu_id
        self.cuda(gpu_id)
        self.gpu_id = gpu_id
        self.gpu = True


class Seq2Seq(nn.Module):
    """Autoencoder of sentences"""

    # `grad_norm` is used to store the gradients of the autoencoder so that they can be
    #  plotted in TensorBoard.
    # It is useful to put it as a class variable because when doing multi-gpu training you can no longer access
    # instance attributes(or at least it is buggy because each instance has a different value for self.grad_norm)
    grad_norm = {}  # purpose: storing the norm of the gradients to force gradients to have the same norm in the gan and the aae
    timer = Timer('AE', enabled=False, log_freq=0, writer=None)

    def __init__(self, emsize, nhidden_enc, nhidden_dec, ntokens, nlayers_enc, nlayers_dec, noise_radius=0.2, tie_weights=False, norm_penalty=None,
                 norm_penalty_threshold=0, hidden_init=False, dropout=0, gpu=False, ngpus=1, gpu_id=None,
                 writer=None, timeit=None, bidirectionnal=False, z_size=None
                 ):
        """

        :param emsize:
        :param nhidden_enc:
        :param nhidden_dec:
        :param ntokens:
        :param nlayers_enc:
        :param nlayers_dec:
        :param noise_radius:
        :param tie_weights:
        :param norm_penalty:
        :param norm_penalty_threshold:
        :param hidden_init:
        :param dropout:
        :param gpu:
        :param ngpus:
        :param gpu_id:
        :param writer:
        :param timeit:
        :param bidirectionnal:
        :param z_size: for the AAE configuration, the encoder output dim must be the same as the dim of the prior
        """
        super(Seq2Seq, self).__init__()
        assert (not bidirectionnal) or (bidirectionnal and (nhidden_enc % 2 == 0))
        self.nhidden_enc = nhidden_enc if not bidirectionnal else nhidden_enc // 2
        self.nhidden_dec = nhidden_dec
        self.bidirectionnal = bidirectionnal
        self.emsize = emsize
        self.ntokens = ntokens
        self.nlayers_enc = nlayers_enc
        self.nlayers_dec = nlayers_dec
        self.noise_radius = noise_radius
        self.hidden_init = hidden_init
        self.dropout = dropout
        self.gpu = gpu
        self.gpu_id = gpu_id
        self.ngpus = ngpus
        self.norm_penalty = norm_penalty
        self.norm_penalty_threshold = norm_penalty_threshold

        self.start_symbols = to_gpu(gpu, Variable(t.ones(10, 1).long()), gpu_id=gpu_id)

        # Vocabulary embedding
        self.embedding = nn.Embedding(ntokens, emsize)
        self.embedding_decoder = nn.Embedding(ntokens, emsize)

        # Embedding dropouts
        if self.dropout is not None:
            self.dropout_enc = nn.Dropout(self.dropout)
            self.dropout_dec = nn.Dropout(self.dropout)

        # RNN Encoder and Decoder
        self.encoder = nn.LSTM(input_size=emsize,
                               hidden_size=self.nhidden_enc,
                               num_layers=nlayers_enc,
                               dropout=dropout,
                               batch_first=True,
                               bidirectional=bidirectionnal)
        if z_size is not None:
            # @todo: what happens if bidirectionnal==True ?
            self.z_size = z_size
            self.linear_enc = nn.Linear(self.nhidden_enc, self.z_size)
        else:
            self.z_size = nhidden_enc
            self.linear_enc = None

        decoder_input_size = emsize + self.z_size
        self.decoder = nn.LSTM(input_size=decoder_input_size,
                               hidden_size=nhidden_dec,
                               num_layers=nlayers_dec,
                               dropout=dropout,
                               batch_first=True)

        # Initialize Linear Transformation
        self.linear = nn.Linear(nhidden_dec, ntokens)
        if tie_weights:
            if emsize == nhidden_dec:
                self.linear.weight = self.embedding_decoder.weight
            else:
                raise ValueError("If you want to tie weights, you need to have emsize=nhidden")

        self.init_weights()

        # if timeit is not None:
        #     Seq2Seq.timer.enabled = True
        #     Seq2Seq.timer.log_freq = timeit
        #     Seq2Seq.timer.writer = writer
        #     # self.forward = Seq2Seq.timer.timeit(self.forward)
        #     self.encode = Seq2Seq.timer.timeit(self.encode)
        #     self.decode = Seq2Seq.timer.timeit(self.decode)

    def init_weights(self):
        initrange = 0.1

        # Initialize Vocabulary Matrix Weight
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding_decoder.weight.data.uniform_(-initrange, initrange)

        # Initialize Encoder and Decoder Weights
        for p in self.encoder.parameters():
            p.data.uniform_(-initrange, initrange)
        for p in self.decoder.parameters():
            p.data.uniform_(-initrange, initrange)

        # Initialize Linear Weight
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)

    def init_hidden(self, bsz):
        zeros1 = Variable(t.zeros(self.nlayers_dec, bsz, self.nhidden_dec))
        zeros2 = Variable(t.zeros(self.nlayers_dec, bsz, self.nhidden_dec))
        return to_gpu(self.gpu, zeros1, gpu_id=self.gpu_id), to_gpu(self.gpu, zeros2, gpu_id=self.gpu_id)

    def init_state(self, bsz):
        zeros = Variable(t.zeros(self.nlayers_dec, bsz, self.nhidden_dec))
        return to_gpu(self.gpu, zeros, gpu_id=self.gpu_id)

    def store_grad_norm(self, grad):
        """
        Monitor gradient norm
        This quantity is used to scale the gradients of the GAN (see train_utils.train_gan_d)
        """
        if self.gpu:
            norm = t.norm(grad, 2, 1)
            Seq2Seq.grad_norm[norm.get_device()] = norm.detach().data.mean()
            return grad
        return

    def forward(self, indices, lengths, noise=True, encode_only=False, keep_hidden=False, return_hidden=False):
        """

        :param indices: integer-encoded sentences. LongTensor
        :param lengths: lengths of the sentences. LongTensor
        :param noise: whether to add Gaussian noise to the code
        :param encode_only: whether to only encode
        :param keep_hidden: whether to store the latent representation to plot its distributional properties in tensorboard
        :return:
        """
        batch_size, maxlen = indices.size()

        hidden = self.encode(indices, lengths, noise, keep_hidden=keep_hidden)
        hidden = hidden if self.linear_enc is None else self.linear_enc.forward(hidden)

        if encode_only:
            return hidden

        if hidden.requires_grad:
            hidden.register_hook(self.store_grad_norm)

        decoded = self.decode(hidden, batch_size, maxlen,
                              indices=indices, lengths=lengths)

        return decoded if not return_hidden else (decoded, hidden)

    def encode(self, indices, lengths, noise, keep_hidden=False):
        """
        :param indices: the integer-encoded sentences. It is a LongTensor
        :param lengths: A list containing the lengths of sentences (they are padded, so the number of columns isn't the length)
                        Note that it could also be a Variable. It actually should be a Variable when you are using multiple GPUs,
                        because pytorch only splits the Variable
        :param noise: whether to add noise to the hidden representation
        :param keep_hidden: whether to store the latent representation to plot its distributional properties in tensorboard
        :return: a latent representation of the sentences, encoded on the unit-sphere
        """
        # `lengths` should be a variable when you use several GPUs, so that the pytorch knows that it should be split
        # among the GPUs you are using
        if isinstance(lengths, t.autograd.Variable):
            lengths_ = lengths.data.cpu().long().numpy().squeeze().tolist()
            if isinstance(lengths_, int):
                lengths_ = [lengths_]
        elif isinstance(lengths, list):
            lengths_ = lengths[:]
        else:
            raise ValueError("Should be either variable or list")

        embeddings = self.embedding(indices)
        if self.dropout is not None:
            embeddings = self.dropout_enc(embeddings)
        packed_embeddings = pack_padded_sequence(input=embeddings, lengths=lengths_, batch_first=True)

        # Encode
        self.encoder.flatten_parameters()
        packed_output, state = self.encoder.forward(packed_embeddings)

        hidden, cell = state
        # batch_size x nhidden
        if not self.bidirectionnal:
            hidden = hidden[-1]  # get hidden state of last layer of encoder
        else:
            hidden = t.cat([hidden[-1], hidden[-2]], -1)

        if self.norm_penalty is None:
            # normalize to unit ball (l2 norm of 1) - p=2, dim=1
            norms = t.norm(hidden, 2, 1)

            # For older versions of PyTorch use:
            # hidden = t.div(hidden, norms.expand_as(hidden))
            # For newest version of PyTorch (as of 8/25) use this:
            hidden = t.div(hidden, norms.unsqueeze(1).expand_as(hidden))

        if noise and self.noise_radius > 0:  # noise to make the task of the discriminator harder in the beginning of training
            gauss_noise = t.normal(means=t.zeros(hidden.size()),
                                   std=self.noise_radius)
            hidden = hidden + to_gpu(self.gpu, Variable(gauss_noise), gpu_id=self.gpu_id)

        if keep_hidden:
            self.hidden = hidden
        return hidden

    def decode(self, hidden, batch_size, maxlen, indices=None, lengths=None):
        # `lengths` should be a variable when you use several GPUs, so that the pytorch knows that it should be split
        # among the GPUs you are using
        if isinstance(lengths, t.autograd.Variable):
            lengths_ = lengths.data.cpu().long().numpy().squeeze().tolist()
            if isinstance(lengths_, int):
                lengths_ = [lengths_]
        elif isinstance(lengths, list):
            lengths_ = lengths[:]
        else:
            raise ValueError("Should be either variable or list")

        # batch x hidden
        all_hidden = hidden.unsqueeze(1).repeat(1, maxlen, 1)

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state = (hidden.unsqueeze(0), self.init_state(batch_size))
        else:
            state = self.init_hidden(batch_size)

        # @todo: remove exposure bias ?
        embeddings = self.embedding_decoder(indices)
        if self.dropout is not None:
            embeddings = self.dropout_dec(embeddings)
        augmented_embeddings = t.cat([embeddings, all_hidden], 2)
        packed_embeddings = pack_padded_sequence(input=augmented_embeddings,
                                                 lengths=lengths_,
                                                 batch_first=True)

        self.decoder.flatten_parameters()
        packed_output, state = self.decoder(packed_embeddings, state)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, maxlen=maxlen) if self.ngpus > 1 else pad_packed_sequence(packed_output, batch_first=True, maxlen=None)

        # reshape to batch_size*maxlen x nhidden before linear over vocab
        decoded = self.linear(output.contiguous().view(-1, self.nhidden_dec))
        decoded = decoded.view(batch_size, output.size(1), self.ntokens)

        return decoded

    def generate(self, hidden, maxlen, sample=True, temp=1.0):
        """
        Generate through decoder; no backprop
        :param hidden: latent code obtained by sampling
        :param sample: whether to sample (vs greedy approach)
        :param temp: temperature parameter in case you sample. The lower, the closer to argmax
        """

        batch_size = hidden.size(0)

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state = (hidden.unsqueeze(0), self.init_state(batch_size))
        else:
            state = self.init_hidden(batch_size)

        # <sos>
        self.start_symbols.data.resize_(batch_size, 1)
        self.start_symbols.data.fill_(1)

        embedding = self.embedding_decoder(self.start_symbols)
        inputs = t.cat([embedding, hidden.unsqueeze(1)], 2)

        # unroll
        all_indices = []
        for i in range(maxlen):
            self.decoder.flatten_parameters()
            output, state = self.decoder(inputs, state)
            overvocab = self.linear(output.squeeze(1))  # batch x |V|

            if not sample:
                vals, indices = t.max(overvocab, 1)  # this is not an error on newer PyTorch
            else:
                # sampling
                probs = F.softmax(overvocab / temp, dim=1)
                indices = t.multinomial(probs, 1)

            all_indices.append(indices)
            embedding = self.embedding_decoder(indices)

            if embedding.dim() == 2:
                inputs = t.cat([embedding.unsqueeze(1), hidden.unsqueeze(1)], 2)
            else:
                inputs = t.cat([embedding, hidden.unsqueeze(1)], 2)
        max_indices = t.stack(all_indices, 1)
        return max_indices

    def keep_gradients(self):
        """
        Store the gradients to plot them in tensorboard. Need to store them because they are erased before training
        the GAN
        """
        self.gradients = {}
        for l in range(self.encoder.num_layers):
            self.gradients['Enc_ih_%d' % l] = getattr(self.encoder, 'weight_ih_l%d' % l).grad.cpu().data.numpy()
            self.gradients['Enc_hh_%d' % l] = getattr(self.encoder, 'weight_hh_l%d' % l).grad.cpu().data.numpy()
        for l in range(self.decoder.num_layers):
            self.gradients['Dec_ih_%d' % l] = getattr(self.decoder, 'weight_ih_l%d' % l).grad.cpu().data.numpy()
            self.gradients['Dec_hh_%d' % l] = getattr(self.decoder, 'weight_hh_l%d' % l).grad.cpu().data.numpy()
        self.gradients['Dec_fc'] = self.linear.weight.grad.cpu().data.numpy()

    def tensorboard(self, writer, n_iter):
        """
        Note that tensorboard default binning (for histograms) is buggy, prefer `doane` instead
        """
        # Weights and gradients
        for l in range(self.encoder.num_layers):
            writer.add_histogram('Enc_ih_w_%d' % l, getattr(self.encoder, 'weight_ih_l%d' % l).data.cpu().numpy(), n_iter, bins='doane')
            writer.add_histogram('Enc_ih_grad_%d' % l, self.gradients['Enc_ih_%d' % l], n_iter, bins='doane')
            writer.add_histogram('Enc_hh_w_%d' % l, getattr(self.encoder, 'weight_hh_l%d' % l).data.cpu().numpy(), n_iter, bins='doane')
            writer.add_histogram('Enc_hh_grad_%d' % l, self.gradients['Enc_hh_%d' % l], n_iter, bins='doane')
        for l in range(self.decoder.num_layers):
            writer.add_histogram('Dec_ih_w_%d' % l, getattr(self.decoder, 'weight_ih_l%d' % l).data.cpu().numpy(), n_iter, bins='doane')
            writer.add_histogram('Dec_ih_grad_%d' % l, self.gradients['Dec_ih_%d' % l], n_iter, bins='doane')
            writer.add_histogram('Dec_hh_w_%d' % l, getattr(self.decoder, 'weight_hh_l%d' % l).data.cpu().numpy(), n_iter, bins='doane')
            writer.add_histogram('Dec_hh_grad_%d' % l, self.gradients['Dec_hh_%d' % l], n_iter, bins='doane')
        writer.add_histogram('Dec_fc_w', self.linear.weight.data.cpu().numpy(), n_iter, bins='doane')
        writer.add_histogram('Dec_fc_grad', self.linear.weight.grad.cpu().data.numpy(), n_iter, bins='doane')
        writer.add_histogram('Dec_fc_grad', self.gradients['Dec_fc'], n_iter, bins='doane')

        # Distributional properties of codes
        trace_cov = np.trace(np.cov(self.hidden.data.cpu().numpy().T))  # @todo: use a bigger number of samples than just the last batch to estimate this ?
        writer.add_scalar('trace_cov_ae', trace_cov, n_iter)
        norm_code = t.mean(self.hidden.norm(2, 1))
        writer.add_scalar('l2_norm_enc', norm_code, n_iter)

    def __cuda__(self, gpu_id=None):
        if gpu_id is None and self.gpu_id is None:
            gpu_id = 0
        if gpu_id is None and self.gpu_id is not None:
            gpu_id = self.gpu_id
        self.cuda(gpu_id)
        self.gpu_id = gpu_id
        self.gpu = True
        self.start_symbols = to_gpu(True, Variable(t.ones(10, 1).long()), gpu_id=gpu_id)
        self.encoder.flatten_parameters()
        self.decoder.flatten_parameters()


def load_models(load_path, old=False):
    """

    :param load_path:
    :param old: True/False.
                Before, the argparse had only one argument --nhidden
                Now it is separated into --nhidden_enc and --nhidden_dec
                For old=True, you load the first
                For old=False, you load the second
    :return:
    """
    model_args = json.load(open("{}/args.json".format(load_path), "r"))
    word2idx = json.load(open("{}/vocab.json".format(load_path), "r"))
    idx2word = {v: k for k, v in word2idx.items()}

    autoencoder = Seq2Seq(emsize=model_args['emsize'],
                          nhidden_enc=model_args['nhidden_enc'] if not old else model_args['nhidden'],
                          nhidden_dec=model_args['nhidden_dec'] if not old else model_args['nhidden'],
                          ntokens=model_args['ntokens'],
                          nlayers=model_args['nlayers'],
                          hidden_init=model_args['hidden_init'],
                          norm_penalty=model_args['norm_penalty'],
                          norm_penalty_threshold=model_args['norm_penalty_threshold'] if 'norm_penalty_threshold' in model_args else None,
                          dropout=model_args['dropout'],
                          bidirectionnal=model_args['bidirectionnal']
                          )
    gan_gen = MLP_G(ninput=model_args['z_size'],
                    noutput=model_args['nhidden_enc'] if not old else model_args['nhidden'],
                    layers=model_args['arch_g'],
                    batchnorm=model_args['bn_gen']
                    )
    gan_disc = MLP_D(ninput=model_args['nhidden_enc'] if not old else model_args['nhidden'],
                     noutput=1,
                     layers=model_args['arch_d'],
                     spectralnorm=model_args['spectralnorm'],
                     batchnorm=model_args['bn_disc'],
                     std_minibatch=model_args['std_minibatch']
                     )

    print('Loading models from' + load_path)
    ae_path = os.path.join(load_path, "autoencoder_model.pt")
    gen_path = os.path.join(load_path, "gan_gen_model.pt")
    disc_path = os.path.join(load_path, "gan_disc_model.pt")

    autoencoder.load_state_dict(t.load(ae_path))
    gan_gen.load_state_dict(t.load(gen_path))
    gan_disc.load_state_dict(t.load(disc_path))
    return model_args, word2idx, idx2word, autoencoder, gan_gen, gan_disc


def load_ae(load_path, old=False):
    """

    :param load_path:
    :param old: True/False.
                Before, the argparse had only one argument --nhidden
                Now it is separated into --nhidden_enc and --nhidden_dec
                For old=True, you load the first
                For old=False, you load the second
    :return:
    """
    model_args = json.load(open("{}/args.json".format(load_path), "r"))
    word2idx = json.load(open("{}/vocab.json".format(load_path), "r"))
    idx2word = {v: k for k, v in word2idx.items()}

    autoencoder = Seq2Seq(emsize=model_args['emsize'],
                          nhidden_enc=model_args['nhidden_enc'] if not old else model_args['nhidden'],
                          nhidden_dec=model_args['nhidden_dec'] if not old else model_args['nhidden'],
                          ntokens=model_args['ntokens'],
                          nlayers_enc=model_args['nlayers_enc'],
                          nlayers_dec=model_args['nlayers_dec'],
                          hidden_init=model_args['hidden_init'],
                          norm_penalty=model_args['norm_penalty'],
                          norm_penalty_threshold=model_args['norm_penalty_threshold'],
                          dropout=model_args['dropout'],
                          bidirectionnal=model_args['bidirectionnal']
                          )

    print('Loading models from' + load_path)
    ae_path = os.path.join(load_path, "autoencoder_model.pt")

    autoencoder.load_state_dict(t.load(ae_path))
    return model_args, word2idx, idx2word, autoencoder


def load_gen(load_path, old=False):
    """

    :param load_path:
    :param old: True/False.
                Before, the argparse had only one argument --nhidden
                Now it is separated into --nhidden_enc and --nhidden_dec
                For old=True, you load the first
                For old=False, you load the second
    :return:
    """
    model_args = json.load(open("{}/args.json".format(load_path), "r"))
    word2idx = json.load(open("{}/vocab.json".format(load_path), "r"))
    idx2word = {v: k for k, v in word2idx.items()}

    gan_gen = MLP_G(ninput=model_args['z_size'],
                    noutput=model_args['nhidden_enc'] if not old else model_args['nhidden'],
                    layers=model_args['arch_g'])

    print('Loading models from' + load_path)
    gen_path = os.path.join(load_path, "gan_gen_model.pt")

    gan_gen.load_state_dict(t.load(gen_path))
    return model_args, word2idx, idx2word, gan_gen


def load_disc(load_path, old=False):
    """

    :param load_path:
    :param old: True/False.
                Before, the argparse had only one argument --nhidden
                Now it is separated into --nhidden_enc and --nhidden_dec
                For old=True, you load the first
                For old=False, you load the second
    :return:
    """
    model_args = json.load(open("{}/args.json".format(load_path), "r"))
    word2idx = json.load(open("{}/vocab.json".format(load_path), "r"))
    idx2word = {v: k for k, v in word2idx.items()}

    gan_disc = MLP_D(ninput=model_args['nhidden_enc'] if not old else model_args['nhidden'],
                     noutput=1,
                     layers=model_args['arch_d'],
                     spectralnorm=model_args['spectralnorm'],
                     batchnorm=model_args['bn_disc'],
                     std_minibatch=model_args['std_minibatch']
                     )

    print('Loading models from' + load_path)
    disc_path = os.path.join(load_path, "gan_disc_model.pt")

    gan_disc.load_state_dict(t.load(disc_path))
    return model_args, word2idx, idx2word, gan_disc


def generate(autoencoder, gan_gen, z, vocab, sample, maxlen):
    """
    Assume noise is batch_size x z_size
    """
    if type(z) == Variable:
        noise = z
    elif type(z) == t.FloatTensor or type(z) == t.cuda.FloatTensor:
        noise = Variable(z, volatile=True)
    elif type(z) == np.ndarray:
        noise = Variable(t.from_numpy(z).float(), volatile=True)
    else:
        raise ValueError("Unsupported input type (noise): {}".format(type(z)))

    gan_gen.eval()
    autoencoder.eval()

    # generate from random noise
    fake_hidden = gan_gen(noise)
    max_indices = autoencoder.generate(hidden=fake_hidden,
                                       maxlen=maxlen,
                                       sample=sample)

    max_indices = max_indices.data.cpu().numpy()
    sentences = []
    for idx in max_indices:
        # generated sentence
        words = [vocab[x] for x in idx]
        # truncate sentences to first occurrence of <eos>
        truncated_sent = []
        for w in words:
            if w != '<eos>':
                truncated_sent.append(w)
            else:
                break
        sent = " ".join(truncated_sent)
        sentences.append(sent)

    return sentences
