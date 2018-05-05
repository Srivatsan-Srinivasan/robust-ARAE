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
from utils import *
from const import *


class LSTM(t.nn.Module):
    def __init__(self, params):
        super(LSTM, self).__init__()
        print("Initializing LSTM")
        self.cuda_flag = params.get('cuda', CUDA_DEFAULT)
        self.model_str = 'LSTM'
        self.params = params

        # Initialize hyperparams.
        self.hidden_dim = params.hidden_dim
        self.batch_size = params.batch_size
        self.embedding_dim = params.emsize
        self.vocab_size = params.ntokens
        self.num_layers = params.num_layers
        self.dropout = params.dropout
        self.embed_dropout = params.embed_dropout
        self.train_embedding = True
        self.tie_weights = False

        # Initialize embeddings.
        additional_dim = self.init_embedding_and_output(embeddings)

        # Initialize network modules.
        self.model_rnn = nn.LSTM(self.embedding_dim + additional_dim, self.hidden_dim, dropout=self.dropout, num_layers=self.num_layers)

        self.hidden = self.init_hidden()
        if self.embed_dropout:
            self.dropout_1 = nn.Dropout(self.dropout)
        self.dropout_2 = nn.Dropout(self.dropout)

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim). The helper function
        # will return torch variable.
        if self.model_str in ['GRU', 'BiGRU']:
            return variable(np.zeros((self.num_layers, self.batch_size, self.hidden_dim)), cuda=self.cuda_flag)
        else:
            return tuple((
                variable(np.zeros((self.num_layers, self.batch_size, self.hidden_dim)), cuda=self.cuda_flag),
                variable(np.zeros((self.num_layers, self.batch_size, self.hidden_dim)), cuda=self.cuda_flag)
            ))

    def init_embedding_and_output(self, embeddings):
        """
        If you tie weights, the embedding and hidden dims should match.
        If they don't, you can add dimensions to the embedding and make only the good part match
        :returns: the additional dim to the embedding
        """
        if self.tie_weights and self.embedding_dim == self.hidden_dim:
            self.word_embeddings = t.nn.Embedding(self.vocab_size, self.embedding_dim)
            self.word_embeddings.weight = nn.Parameter(embeddings, requires_grad=self.train_embedding)
            self.hidden2out = nn.Linear(self.hidden_dim, self.output_size)
            self.hidden2out.weight = nn.Parameter(embeddings, requires_grad=True)
            return 0
        elif self.tie_weights and self.embedding_dim > self.hidden_dim:
            raise ValueError("self.embedding_dim should be bigger than self.hidden_dim")
        elif self.tie_weights and self.embedding_dim < self.hidden_dim:
            additional_dim = self.hidden_dim - self.embedding_dim
            self.word_embeddings = t.nn.Embedding(self.vocab_size, self.embedding_dim + additional_dim)
            embedding_weights = t.cat([embeddings,
                                       self.word_embeddings.weight.data[:, self.embedding_dim:]],
                                      1)
            self.word_embeddings.weight = nn.Parameter(embedding_weights, requires_grad=self.train_embedding)
            self.hidden2out = nn.Linear(self.hidden_dim, self.output_size)
            hidden_weights = t.cat([embeddings,
                                    self.hidden2out.weight.data[:, self.embedding_dim:]],
                                   1)
            self.hidden2out.weight = nn.Parameter(hidden_weights, requires_grad=self.train_embedding)
            return additional_dim
        elif not self.tie_weights:
            self.word_embeddings = t.nn.Embedding(self.vocab_size, self.embedding_dim)
            self.word_embeddings.weight = nn.Parameter(embeddings, requires_grad=self.train_embedding)
            self.hidden2out = nn.Linear(self.hidden_dim, self.output_size)
            return 0
        else:
            raise ValueError("One of the conditions before should have been True. Problem in the code.")

    def forward(self, x_batch, debug=False):
        if debug:
            import pdb
            pdb.set_trace()

        # EMBEDDING
        embeds = self.word_embeddings(x_batch)
        # going from ` batch_size x bptt_length x embed_dim` to `bptt_length x batch_size x embed_dim`
        embeds = embeds.permute(1, 0, 2)
        if self.embed_dropout:
            embeds = self.dropout_1(embeds)

        # RECURRENT
        rnn_out, self.hidden = self.model_rnn(embeds, self.hidden)
        rnn_out = rnn_out.permute(1, 0, 2)
        rnn_out = self.dropout_2(rnn_out)

        # OUTPUT
        out_linear = self.hidden2out(rnn_out)
        return out_linear, self.hidden


class TemporalCrossEntropyLoss(t.nn.modules.loss._WeightedLoss):
    r"""This criterion combines `LogSoftMax` and `NLLLoss` in one single class.

    It is useful when training a temporal classification problem with `C` classes over time series of length `T`.
    If provided, the optional argument `weight` should be a 1D `Tensor`
    assigning weight to each of the classes.
    This is particularly useful when you have an unbalanced training set.

    The `input` is expected to contain scores for each class.

    `input` has to be a 3D `Tensor` of size `(minibatch, C, T)`.

    This criterion expects a class index (0 to C-1) as the
    `target` for each value of a 2D tensor of size `(minibatch, T)`

    The loss can be described as, for each time step `t`::

        loss(x, class, t) = -log(exp(x[class]) / (\sum_j exp(x[j])))
                       = -x[class] + log(\sum_j exp(x[j]))
    The total loss being:
        loss(x, class) = \sum_t loss(x, class, t)


    or in the case of the `weight` argument being specified::

        loss(x, class, t) = weight[class] * (-x[class] + log(\sum_j exp(x[j])))

    The losses are averaged across observations for each minibatch.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each class.
           If given, has to be a Tensor of size "C"
        size_average (bool, optional): By default, the losses are averaged over observations for each minibatch.
           However, if the field size_average is set to ``False``, the losses are
           instead summed for each minibatch. Ignored if reduce is ``False``.
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When size_average is
            ``True``, the loss is averaged over non-ignored targets.

    Shape:
        - Input: :math:`(N, C, T)` where `C = number of classes` and `T = sentence length`
        - Target: :math:`(N, T)` where each value is `0 <= targets[i] <= C-1`
        - Output: scalar. If reduce is ``False``, then :math:`(N)` instead.

    Examples::

        >>> loss = nn.CrossEntropyLoss2D()
        >>> input = variable(torch.randn(batch_size, vocab_size, sentence_length), requires_grad=True)  # for each element of the batch, for each position x=0..sentence_length-1, it gives a probability distribution over the |V| possible words
        >>> target = variable(...)  # size (batch_size, sentence_length). LongTensor containing the correct class (correct next word) at each position of the sentence
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self, weight=None, size_average=True, ignore_index=-100):
        """Average over the batch_size"""
        super(TemporalCrossEntropyLoss, self).__init__(weight, size_average)
        self.ignore_index = ignore_index
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, size_average=size_average, ignore_index=ignore_index)

    def forward(self, pred, true):
        """
        Let `C` be the number of classes and `|V|` the vocab size
        What this class does is just reshaping the inputs so that you can use classical cross entropy on top of that

        :param pred: FloatTensor of shape (batch_size, |V|, C)
        :param true: LongTensor of shape (batch_size, C)
        :return:
        """
        t.nn.modules.loss._assert_no_grad(true)

        # doing it this way allows to use parallelism. Better than looping on last dim !
        # note that this version of pytorch seems outdated
        true_ = true.contiguous().view(true.size(0) * true.size(1))  # true.size() = (batch_size, bptt_length)
        pred_ = pred.contiguous().view(pred.size(0) * pred.size(2), pred.size(1))  # pred.size() = (batch_size, vocab_size, bptt_length)
        return self.cross_entropy.forward(pred_, true_)
