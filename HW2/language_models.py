# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 15:46:07 2018

@author: SrivatsanPC
"""

import torch as t
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import os

os.chdir('../HW2')  # so that there is not any import bug in case HW2 is not already the working directory
from utils import *
from const import *


class LSTM(t.nn.Module):
    def __init__(self, params, embeddings, cuda=CUDA_DEFAULT):
        super(LSTM, self).__init__()
        self.cuda = cuda

        # Initialize hyperparams.
        self.hidden_dim = params.get('hidden_dim', default=100)
        self.batch_size = params.get('batch_size', default=32)
        self.embedding_dim = params.get('embedding_dim', default=300)
        self.vocab_size = params.get('vocab_size', default=1000)
        self.output_size = params.get('output_size', default=self.vocab_size)
        self.num_layers = params.get('num_layers', default=1)
        self.dropout = params.get('dropout', default=0.5)

        # Initialize embeddings. Static embeddings for now.
        self.word_embeddings = t.nn.Embedding(self.vocab_size, self.embedding_size)
        self.word_embeddings.weight = nn.Parameter(embeddings, requires_grad=False)

        # Initialize networks.
        self.rnn = nn.LSTM(self.embedding_dim, self.hidden_dim, dropout=self.dropout)
        self.hidden2out = nn.Linear(self.hidden_dim, self.output_size)
        self.hidden = self.init_hidden(self.num_layers)

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim). The helper function
        # will return torch variable.
        return (variable(np.zeros((self.num_layers, self.batch_size, self.hidden_dim)), cuda=self.cuda, requires_grad=True))

    def forward(self, x_batch):
        embeds = self.word_embeddings(x_batch)
        rnn_out, self.hidden = self.rnn(embeds, self.hidden_dim)

        # Need to train it once and check the output to match dimensions.
        # Won't work in the present state.
        out_linear = self.hidden2out(rnn_out.view())

        # Use cross entropy loss on it directly.
        return out_linear


class GRU(LSTM):
    def __init__(self, params, embeddings, cuda=False):
        LSTM.__init__(self, params, embeddings, cuda=cuda)
        self.rnn = nn.GRU(self.embedding_dim, self.hidden_dim, dropout=self.dropout)


class BiGRU(LSTM):
    def __init__(self, params, embeddings, cuda=False):
        LSTM.__init__(self, params, embeddings, cuda=cuda)
        self.rnn = nn.GRU(self.embedding_dim, self.hidden_dim, dropout=self.dropout, bidirectional=True)


class BiLSTM(LSTM):
    def __init__(self, params, embeddings, cuda=False):
        LSTM.__init__(self, params, embeddings, cuda=cuda)
        self.rnn = nn.LSTM(self.embedding_dim, self.hidden_dim, dropout=self.dropout, bidirectional=True)


class NNLM(t.nn.Module):
    """
    Model defined in 'A Neural Probabilistic Language Model'
    It is implemented using convolutions instead of linear layers because of the format of the data. This way it doesn't involve more pre-processing

    However it makes shuffling impossible, which is a problem for SGD (breaks the iid assumption)
    """
    def __init__(self, context_size, embeddings, train_embedding=False):
        super(NNLM, self).__init__()
        self.context_size = context_size
        self.vocab_size = embeddings.size(0)
        self.embed_dim = embeddings.size(1)

        self.w = t.nn.Embedding(self.vocab_size, self.embed_dim)
        self.w.weight = t.nn.Parameter(embeddings, requires_grad=train_embedding)

        self.conv = t.nn.Conv1d(self.embed_dim, self.vocab_size, context_size)

    def forward(self, x):
        xx = self.w(x).transpose(2, 1)
        xx = self.conv(xx)
        return xx[:, :, :-1]  # you don't take into account the last predictions that is actually the prediction of the first word of the next batch

    @staticmethod
    def criterion(pred, true):
        """
        pred is a batch_size x |V| x T tensor
        true is a batch_size x T tensor

        The output of the model is a distribution probability for each word in the batch
        It makes batch_size x T class prediction
        Therefore you need to apply the cross_entropy loss on each column t=1..T
        You can see it as T classification outputs at the same time
        """
        l = 0
        for k in range(pred.size(2)):  # one cross entropy per column
            pred_ = pred[:, :, k:k + 1].squeeze()
            true_ = true[:, k:k + 1].squeeze()
            l += F.cross_entropy(pred_, true_)
        return l / pred.size(2)
