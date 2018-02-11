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
    def __init__(self, params, embeddings):
        super(LSTM, self).__init__()
        self.cuda_flag = params.get('cuda', CUDA_DEFAULT)
        self.model_str = 'LSTM'

        # Initialize hyperparams.
        self.hidden_dim = params.get('hidden_dim', 100)
        self.batch_size = params.get('batch_size', 32)
        self.embedding_dim = params.get('embedding_dim', 300)
        self.vocab_size = params.get('vocab_size', 1000)
        self.output_size = params.get('output_size',self.vocab_size)
        self.num_layers = params.get('num_layers', 1)
        self.dropout = params.get('dropout', 0.5)
        self.train_embedding = params.get('train_embedding',False)

        # Initialize embeddings. Static embeddings for now.
        self.word_embeddings = t.nn.Embedding(self.vocab_size, self.embedding_dim)
        self.word_embeddings.weight = nn.Parameter(embeddings, requires_grad = self.train_embedding)

        # Initialize network modules.
        self.model_rnn  = nn.LSTM(self.embedding_dim, self.hidden_dim, dropout=self.dropout, num_layers = self.num_layers)
        self.hidden2out = nn.Linear(self.hidden_dim, self.output_size)
        self.hidden     = self.init_hidden()
        self.dropout_1  = nn.Dropout(self.dropout)

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim). The helper function
        # will return torch variable.
        if self.model_str in ['GRU', 'BiGRU']:
            return variable(np.zeros((self.num_layers, self.batch_size, self.hidden_dim)), cuda=self.cuda_flag, requires_grad=True)
        else:
            return tuple((
                    variable(np.zeros((self.num_layers, self.batch_size, self.hidden_dim)), cuda=self.cuda_flag, requires_grad=True),
                    variable(np.zeros((self.num_layers, self.batch_size, self.hidden_dim)), cuda=self.cuda_flag, requires_grad=True)
                   ))           

    def forward(self, x_batch):
        embeds               = self.word_embeddings(x_batch)
        dim1, dim2           = x_batch.size()[1], x_batch.size()[0]
        rnn_out, self.hidden = self.model_rnn(embeds.view(dim1,dim2,-1), self.hidden)    
        #Based on Yoon's advice - dropout before projecting on linear layer.
        self.dropout_1(rnn_out)        
        out_linear           = self.hidden2out(rnn_out.view(dim1,dim2,-1))
        return out_linear


class GRU(LSTM):
    def __init__(self, params, embeddings):
        LSTM.__init__(self, params, embeddings)
        self.model_str = 'GRU'
        self.model_rnn = nn.GRU(self.embedding_dim, self.hidden_dim, dropout=self.dropout, num_layers = self.num_layers)


class BiGRU(LSTM):
    def __init__(self, params, embeddings):
        LSTM.__init__(self, params, embeddings)
        self.model_str = 'BiGRU'
        self.model_rnn = nn.GRU(self.embedding_dim, self.hidden_dim, dropout=self.dropout, num_layers = self.num_layers, bidirectional=True)


class BiLSTM(LSTM):
    def __init__(self, params, embeddings):
        LSTM.__init__(self, params, embeddings)
        self.model_str = 'BiLSTM'
        self.model_rnn = nn.LSTM(self.embedding_dim, self.hidden_dim, dropout=self.dropout, bidirectional=True, num_layers = self.num_layers)


class NNLM(t.nn.Module):
    """
    Model defined in 'A Neural Probabilistic Language Model'
    It is implemented using convolutions instead of linear layers because of the format of the data. This way it doesn't involve more pre-processing

    However it makes shuffling impossible, which is a problem for SGD (breaks the iid assumption)
    """
    def __init__(self, params, embeddings):
        super(NNLM, self).__init__()
        self.model_str = 'NNLM'
        self.context_size = int(params.get('context_size'))
        self.train_embedding = params.get('train_embedding', False)
        self.vocab_size = embeddings.size(0)
        self.embed_dim = embeddings.size(1)

        self.w = t.nn.Embedding(self.vocab_size, self.embed_dim)
        self.w.weight = t.nn.Parameter(embeddings, requires_grad = self.train_embedding )

        # self.dropout = t.nn.Dropout()
        self.conv1 = t.nn.Conv1d(self.embed_dim, self.vocab_size, self.context_size)
        self.conv2 = t.nn.Conv1d(self.embed_dim, self.vocab_size, self.context_size)

    def forward(self, x):
        xx = self.w(x).transpose(2, 1)
        # xx = self.dropout(xx)
        xx1 = F.tanh(self.conv1(xx))
        xx2 = self.conv2(xx)
        xx = xx1 + xx2
        return xx[:, :, :-1]  # you don't take into account the last predictions that is actually the prediction of the first word of the next batch


class NNLM2(t.nn.Module):
    """
    Model defined in 'A Neural Probabilistic Language Model'
    It is implemented using a linear
    """
    def __init__(self, params, embeddings):
        super(NNLM2, self).__init__()
        self.model_str = 'NNLM2'
        self.context_size = int(params.get('context_size'))
        self.train_embedding = params.get('train_embedding', False)
        self.vocab_size = embeddings.size(0)
        self.embed_dim = embeddings.size(1)

        self.w = t.nn.Embedding(self.vocab_size, self.embed_dim)
        self.w.weight = t.nn.Parameter(embeddings, requires_grad=self.train_embedding)

        self.dropout = t.nn.Dropout()
        self.fc1 = t.nn.Linear(self.embed_dim*self.context_size, 50)
        self.fc11 = t.nn.Linear(50, self.vocab_size, bias=False)
        # self.fc2 = t.nn.Linear(self.embed_dim*self.context_size, self.vocab_size)

    def forward(self, x):
        xx = self.w(x)
        # xx = self.dropout(xx)
        xx = xx.contiguous().view(xx.size(0), -1)  # .contiguous() because .view() requires the tensor to be stored in contiguous memory blocks
        xx = F.tanh(self.fc1(xx))
        xx = self.dropout(xx)
        xx = self.fc11(xx)
        return xx


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
        true_ = true.contiguous().view(true.size(0)*true.size(1))
        pred_ = pred.contiguous().view(pred.size(0)*pred.size(2), pred.size(1))
        return self.cross_entropy.forward(pred_, true_)
