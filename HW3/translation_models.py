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
    """
    Implementation of `Sequence to Sequence Learning with Neural Networks`
    https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
    """
    def __init__(self, params, source_embeddings, target_embeddings):
        super(LSTM, self).__init__()
        print("Initializing LSTM")
        self.cuda_flag = params.get('cuda', CUDA_DEFAULT)
        self.model_str = 'LSTM'
        self.params = params

        # Initialize hyperparams.
        self.hidden_dim = params.get('hidden_dim', 100)
        self.batch_size = params.get('batch_size', 32)
        try:
            # if you provide pre-trained embeddings for target/source, they should have the same embedding dim
            assert source_embeddings.size(1) == target_embeddings.size(1)
            self.embedding_dim = source_embeddings.size(1)
            self.vocab_size = target_embeddings.size(0)
        except:
            # if you dont provide a pre-trained embedding, you have to provide these
            self.embedding_dim = params.get('embedding_dim')  # @todo: add this to the argparse
            self.vocab_size = params.get('vocab_size')  # @todo: add this to the argparse
            assert self.embedding_dim is not None and self.vocab_size is not None
        self.output_size = params.get('output_size', self.vocab_size)
        self.num_layers = params.get('num_layers', 1)
        self.dropout = params.get('dropout', 0.5)
        self.embed_dropout = params.get('embed_dropout')
        self.train_embedding = params.get('train_embedding', False)

        # Initialize embeddings. Static embeddings for now.
        self.source_embeddings = t.nn.Embedding(self.vocab_size, self.embedding_dim)
        self.target_embeddings = t.nn.Embedding(self.vocab_size, self.embedding_dim)
        if source_embeddings is not None:
            self.source_embeddings.weight = nn.Parameter(source_embeddings, requires_grad=self.train_embedding)
        if target_embeddings is not None:
            self.target_embeddings.weight = nn.Parameter(target_embeddings, requires_grad=self.train_embedding)

        # Initialize network modules.
        self.encoder_rnn = nn.LSTM(self.embedding_dim, self.hidden_dim, dropout=self.dropout, num_layers=self.num_layers)
        self.decoder_rnn = nn.LSTM(self.embedding_dim + self.hidden_dim, self.hidden_dim, dropout=self.dropout, num_layers=self.num_layers)
        self.hidden2out = nn.Linear(self.hidden_dim, self.output_size)
        self.hidden_enc = self.init_hidden()
        self.hidden_dec = self.init_hidden()
        if self.embed_dropout:
            self.dropout_1s = nn.Dropout(self.dropout)
            self.dropout_1t = nn.Dropout(self.dropout)
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

    def forward(self, x_source, x_target, debug=False):
        """
        :param x_source: the source sentence
        :param x_target: the target (translated) sentence
        :param debug:
        :return:
        """
        if debug:
            import pdb
            pdb.set_trace()

        # EMBEDDING
        xx_source = self.reverse_source(x_source)
        embedded_x_source = self.source_embeddings(xx_source)
        embedded_x_target = self.source_embeddings(x_target)
        if self.embed_dropout:
            embedded_x_source = self.dropout_1s(embedded_x_source)
            embedded_x_target = self.dropout_1t(embedded_x_target)

        # ENCODING SOURCE SENTENCE INTO FIXED LENGTH VECTOR
        _, self.hidden_enc = self.encoder_rnn(embedded_x_source, self.hidden_enc)

        # DECODING
        embedded_x_target = self.append_hidden_to_target(embedded_x_target)
        rnn_out, self.hidden_dec = self.decoder_rnn(embedded_x_target, self.hidden_dec)
        rnn_out = self.dropout_2(rnn_out)
        
        # OUTPUT
        out_linear = self.hidden2out(rnn_out)
        return out_linear, self.hidden

    # @todo: implement this
    def reverse_source(self, x):
        """Reverse the source sentence x. Empirically it was observed to work better in terms of final valid PPL, and especially for long sentences"""
        pass

    # @todo: implement this
    def append_hidden_to_target(self, x):
        """Append self.hidden_enc to all timesteps of x"""
        pass


class LSTMA(t.nn.Module):
    """
    Implementation of `Neural Machine Translation by Jointly Learning to Align and Translate`
    https://arxiv.org/abs/1409.0473
    """
    def __init__(self, params, source_embeddings, target_embeddings):
        super(LSTMA, self).__init__()
        print("Initializing LSTMA")
        self.cuda_flag = params.get('cuda', CUDA_DEFAULT)
        self.model_str = 'LSTMA'
        self.params = params

        # Initialize hyperparams.
        self.hidden_dim = params.get('hidden_dim', 100)
        self.batch_size = params.get('batch_size', 32)
        try:
            # if you provide pre-trained embeddings for target/source, they should have the same embedding dim
            assert source_embeddings.size(1) == target_embeddings.size(1)
            self.embedding_dim = source_embeddings.size(1)
            self.vocab_size = target_embeddings.size(0)
        except:
            # if you dont provide a pre-trained embedding, you have to provide these
            self.embedding_dim = params.get('embedding_dim')  # @todo: add this to the argparse
            self.vocab_size = params.get('vocab_size')  # @todo: add this to the argparse
            assert self.embedding_dim is not None and self.vocab_size is not None
        self.output_size = params.get('output_size', self.vocab_size)
        self.num_layers = params.get('num_layers', 1)
        self.dropout = params.get('dropout', 0.5)
        self.embed_dropout = params.get('embed_dropout')
        self.train_embedding = params.get('train_embedding', False)

        # Initialize embeddings. Static embeddings for now.
        self.source_embeddings = t.nn.Embedding(self.vocab_size, self.embedding_dim)
        self.target_embeddings = t.nn.Embedding(self.vocab_size, self.embedding_dim)
        if source_embeddings is not None:
            self.source_embeddings.weight = nn.Parameter(source_embeddings, requires_grad=self.train_embedding)
        if target_embeddings is not None:
            self.target_embeddings.weight = nn.Parameter(target_embeddings, requires_grad=self.train_embedding)

        # Initialize network modules.
        # note that the encoder is a BiLSTM. The output is modified by the fact that the hidden dim is doubled, and if you set
        # the number of layers to L, there will actually be 2L layers (the forward ones and the backward ones). Consequently the first
        # dimension of the hidden outputs of the forward pass (the 2nd output in the tuple) will be a tuple of
        # 2 tensors having as first dim twice the hidden dim you set
        self.encoder_rnn = nn.LSTM(self.embedding_dim, self.hidden_dim//2, dropout=self.dropout, num_layers=self.num_layers, bidirectional=True, batch_first=True)
        self.decoder_rnn = nn.LSTM(self.embedding_dim + self.hidden_dim, self.hidden_dim, dropout=self.dropout, num_layers=self.num_layers, batch_first=True)
        self.hidden_dec_initializer = nn.Linear(self.hidden_dim // 2, self.hidden_dim)
        self.hidden2out = nn.Linear(self.hidden_dim*2, self.output_size)
        if self.embed_dropout:
            self.dropout_1s = nn.Dropout(self.dropout)
            self.dropout_1t = nn.Dropout(self.dropout)
        self.dropout_2 = nn.Dropout(self.dropout)

    def init_hidden(self, data, type):
        """
        Initialize the hidden state, either for the encoder or the decoder

        For type=`enc`, it should just be initialized with 0s
        For type=`dec`, it should be initialized with tanh(W h1_backward) (see page 13 of the paper, last paragraph)
        """
        if type == 'dec':
            # in that case is the output of the encoder
            return (
                F.tanh(self.hidden_dec_initializer(data[:, 0, self.hidden_dim // 2:])),  # @todo: verify that the last hdim/2 weights actually correspond to the backward layer(s)
                variable(np.zeros((self.num_layers, self.batch_size, self.hidden_dim)), cuda=self.cuda_flag)
            )
        elif type == 'enc':
            # in that case data is None
            return tuple((
                variable(np.zeros((self.num_layers, self.batch_size, self.hidden_dim)), cuda=self.cuda_flag),
                variable(np.zeros((self.num_layers, self.batch_size, self.hidden_dim)), cuda=self.cuda_flag)
            ))
        else:
            raise ValueError('the type should be either `dec` or `enc`')

    def forward(self, x_source, x_target, debug=False):
        if debug:
            import pdb
            pdb.set_trace()

        # EMBEDDING
        embedded_x_source = self.source_embeddings(x_source)
        embedded_x_target = self.target_embeddings(x_target[:, :-1])  # don't make a prediction for the word following the last one
        if self.embed_dropout:
            embedded_x_source = self.dropout_1s(embedded_x_source)
            embedded_x_target = self.dropout_1t(embedded_x_target)

        # RECURRENT
        hidden = self.init_hidden(None, 'enc')
        enc_out, _ = self.encoder_rnn(embedded_x_source, hidden)
        hidden = self.init_hidden(enc_out, 'dec')
        dec_out, _ = self.decoder_rnn(embedded_x_target, hidden)

        # ATTENTION
        scores = t.bmm(enc_out, dec_out.transpose(1, 2))  # this will be a batch x source_len x target_len
        attn_dist = F.softmax(scores.permute(1,0,2)).permute(1,0,2)  # batch x source_len x target_len
        context = t.bmm(attn_dist.permute(0,2,1), enc_out)  # batch x target_len x hidden_dim

        # OUTPUT
        # concatenate the output of the decoder and the context and apply nonlinearity
        pred = F.tanh(t.cat([dec_out, context], -1))  # @todo : tanh necessary ?
        pred = self.dropout_2(pred)  # batch x target_len x 2 hdim
        pred = self.hidden2out(pred)
        return pred


# @ todo: is it useful for this assignment ?
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
