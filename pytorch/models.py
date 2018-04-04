import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import to_gpu
import json
import os
import numpy as np


class MLP_D(nn.Module):
    def __init__(self, ninput, noutput, layers, activation=nn.LeakyReLU(0.2), gpu=False, weight_init='default',
                 std_minibatch=True, batchnorm=False):
        super(MLP_D, self).__init__()
        self.ninput = ninput
        self.noutput = noutput
        self.std_minibatch = std_minibatch
        if isinstance(activation, t.nn.ReLU):
            self.negative_slope = 0
        elif isinstance(activation, t.nn.LeakyReLU):
            self.negative_slope = activation.negative_slope
        else:
            raise ValueError('Not implemented')

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)

            # No batch normalization after first layer
            if i != 0 and batchnorm:
                bn = nn.BatchNorm1d(layer_sizes[i+1], eps=1e-05, momentum=0.1)
                self.layers.append(bn)
                self.add_module("bn"+str(i+1), bn)

            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)

        layer = nn.Linear(layer_sizes[-1]+int(std_minibatch), noutput)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)

        self.init_weights(weight_init)

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
        layer = self.layers[-1]

        if self.std_minibatch:
            x_std_feature = t.mean(t.std(x, 0))
            x = t.cat([x, x_std_feature], 1)

        x = layer(x)
        x = t.mean(x)
        return x

    def init_weights(self, weight_init='default'):
        if weight_init == 'default':
            init_std = 0.02
            for layer in self.layers:
                try:
                    layer.weight.data.normal_(0, init_std)
                    layer.bias.data.fill_(0)
                except:
                    pass
        elif weight_init == 'he':
            for layer in self.layers:
                try:
                    t.nn.init.kaiming_normal_(layer.weight.data, a=self.negative_slope)
                    layer.bias.data.fill_(0)
                except:
                    pass
        else:
            raise NotImplementedError


class MLP_G(nn.Module):
    def __init__(self, ninput, noutput, layers, activation=nn.ReLU(), gpu=False, weight_init='default', batchnorm=True):
        super(MLP_G, self).__init__()
        self.ninput = ninput
        self.noutput = noutput
        if isinstance(activation, t.nn.ReLU):
            self.negative_slope = 0
        elif isinstance(activation, t.nn.LeakyReLU):
            self.negative_slope = activation.negative_slope
        else:
            raise ValueError('Not implemented')

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)

            if batchnorm:
                bn = nn.BatchNorm1d(layer_sizes[i+1], eps=1e-05, momentum=0.1)
                self.layers.append(bn)
                self.add_module("bn"+str(i+1), bn)

            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)

        self.init_weights(weight_init)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def init_weights(self, weight_init='default'):
        if weight_init == 'default':
            init_std = 0.02
            for layer in self.layers:
                try:
                    layer.weight.data.normal_(0, init_std)
                    layer.bias.data.fill_(0)
                except:
                    pass
        elif weight_init == 'he':
            for layer in self.layers:
                try:
                    t.nn.init.kaiming_normal_(layer.weight.data, a=self.negative_slope)
                    layer.bias.data.fill_(0)
                except:
                    pass
        else:
            raise NotImplementedError('Not implemented')


class Seq2Seq(nn.Module):
    def __init__(self, emsize, nhidden, ntokens, nlayers, noise_radius=0.2,
                 hidden_init=False, dropout=0, gpu=False):
        super(Seq2Seq, self).__init__()
        self.nhidden = nhidden
        self.emsize = emsize
        self.ntokens = ntokens
        self.nlayers = nlayers
        self.noise_radius = noise_radius
        self.hidden_init = hidden_init
        self.dropout = dropout
        self.gpu = gpu

        self.start_symbols = to_gpu(gpu, Variable(t.ones(10, 1).long()))

        # Vocabulary embedding
        self.embedding = nn.Embedding(ntokens, emsize)
        self.embedding_decoder = nn.Embedding(ntokens, emsize)

        # RNN Encoder and Decoder
        self.encoder = nn.LSTM(input_size=emsize,
                               hidden_size=nhidden,
                               num_layers=nlayers,
                               dropout=dropout,
                               batch_first=True)

        decoder_input_size = emsize+nhidden
        self.decoder = nn.LSTM(input_size=decoder_input_size,
                               hidden_size=nhidden,
                               num_layers=1,
                               dropout=dropout,
                               batch_first=True)

        # Initialize Linear Transformation
        self.linear = nn.Linear(nhidden, ntokens)

        self.init_weights()

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
        zeros1 = Variable(t.zeros(self.nlayers, bsz, self.nhidden))
        zeros2 = Variable(t.zeros(self.nlayers, bsz, self.nhidden))
        return (to_gpu(self.gpu, zeros1), to_gpu(self.gpu, zeros2))

    def init_state(self, bsz):
        zeros = Variable(t.zeros(self.nlayers, bsz, self.nhidden))
        return to_gpu(self.gpu, zeros)

    def store_grad_norm(self, grad):
        """Monitor gradient norm"""
        norm = t.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

    def forward(self, indices, lengths, noise, encode_only=False):
        """

        :param indices: integer-encoded sentences. LongTensor
        :param lengths: lengths of the sentences. LongTensor
        :param noise: whether to add Gaussian noise to the code
        :param encode_only: whether to only encode
        :return:
        """
        batch_size, maxlen = indices.size()

        hidden = self.encode(indices, lengths, noise)

        if encode_only:
            return hidden

        if hidden.requires_grad:
            hidden.register_hook(self.store_grad_norm)

        decoded = self.decode(hidden, batch_size, maxlen,
                              indices=indices, lengths=lengths)

        return decoded

    def encode(self, indices, lengths, noise):
        """
        :param indices: the integer-encoded sentences. It is a LongTensor
        :param lengths: A LongTensor containing the lengths of sentences (they are padded, so the number of columns isn't the length)
        :param noise: whether to add noise to the hidden representation
        :return: a latent representation of the sentences, encoded on the unit-sphere
        """
        embeddings = self.embedding(indices)
        packed_embeddings = pack_padded_sequence(input=embeddings,
                                                 lengths=lengths,
                                                 batch_first=True)

        # Encode
        packed_output, state = self.encoder(packed_embeddings)

        hidden, cell = state
        # batch_size x nhidden
        hidden = hidden[-1]  # get hidden state of last layer of encoder

        # normalize to unit ball (l2 norm of 1) - p=2, dim=1
        norms = t.norm(hidden, 2, 1)
        
        # For older versions of PyTorch use:
        # hidden = t.div(hidden, norms.expand_as(hidden))
        # For newest version of PyTorch (as of 8/25) use this:
        hidden = t.div(hidden, norms.unsqueeze(1).expand_as(hidden))

        # @todo: why noise ?
        if noise and self.noise_radius > 0:
            gauss_noise = t.normal(means=t.zeros(hidden.size()),
                                       std=self.noise_radius)
            hidden = hidden + to_gpu(self.gpu, Variable(gauss_noise))

        return hidden

    def decode(self, hidden, batch_size, maxlen, indices=None, lengths=None):
        # batch x hidden
        all_hidden = hidden.unsqueeze(1).repeat(1, maxlen, 1)

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state = (hidden.unsqueeze(0), self.init_state(batch_size))
        else:
            state = self.init_hidden(batch_size)

        # @todo: exposure bias ?
        embeddings = self.embedding_decoder(indices)
        augmented_embeddings = t.cat([embeddings, all_hidden], 2)
        packed_embeddings = pack_padded_sequence(input=augmented_embeddings,
                                                 lengths=lengths,
                                                 batch_first=True)

        packed_output, state = self.decoder(packed_embeddings, state)
        output, lengths = pad_packed_sequence(packed_output, batch_first=True)

        # reshape to batch_size*maxlen x nhidden before linear over vocab
        decoded = self.linear(output.contiguous().view(-1, self.nhidden))
        decoded = decoded.view(batch_size, maxlen, self.ntokens)

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
            output, state = self.decoder(inputs, state)
            overvocab = self.linear(output.squeeze(1))

            if not sample:
                vals, indices = t.max(overvocab, 1)  # this is not an error on newer PyTorch
            else:
                # sampling
                probs = F.softmax(overvocab/temp)
                indices = t.multinomial(probs, 1)

            all_indices.append(indices)
            embedding = self.embedding_decoder(indices)

            if embedding.dim() == 2:
                inputs = t.cat([embedding.unsqueeze(1), hidden.unsqueeze(1)], 2)
            else:
                inputs = t.cat([embedding, hidden.unsqueeze(1)], 2)
        max_indices = t.stack(all_indices, 1)
        return max_indices


def load_models(load_path):
    model_args = json.load(open("{}/args.json".format(load_path), "r"))
    word2idx = json.load(open("{}/vocab.json".format(load_path), "r"))
    idx2word = {v: k for k, v in word2idx.items()}

    autoencoder = Seq2Seq(emsize=model_args['emsize'],
                          nhidden=model_args['nhidden'],
                          ntokens=model_args['ntokens'],
                          nlayers=model_args['nlayers'],
                          hidden_init=model_args['hidden_init'])
    gan_gen = MLP_G(ninput=model_args['z_size'],
                    noutput=model_args['nhidden'],
                    layers=model_args['arch_g'])
    gan_disc = MLP_D(ninput=model_args['nhidden'],
                     noutput=1,
                     layers=model_args['arch_d'])

    print('Loading models from'+load_path)
    ae_path = os.path.join(load_path, "autoencoder_model.pt")
    gen_path = os.path.join(load_path, "gan_gen_model.pt")
    disc_path = os.path.join(load_path, "gan_disc_model.pt")

    autoencoder.load_state_dict(t.load(ae_path))
    gan_gen.load_state_dict(t.load(gen_path))
    gan_disc.load_state_dict(t.load(disc_path))
    return model_args, idx2word, autoencoder, gan_gen, gan_disc


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
