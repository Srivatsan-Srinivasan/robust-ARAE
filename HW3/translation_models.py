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

    NOTE THAT ITS INPUT SHOULD HAVE THE BATCH SIZE FIRST !!!!!
    """

    def __init__(self, params, source_embeddings=None, target_embeddings=None):
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
            self.source_vocab_size = params.get('source_vocab_size')
            self.target_vocab_size = params.get('target_vocab_size')
            assert source_embeddings.size(1) == target_embeddings.size(1)
            self.embedding_dim = source_embeddings.size(1)
        except:
            # if you dont provide a pre-trained embedding, you have to provide these
            self.source_vocab_size = params.get('source_vocab_size')
            self.target_vocab_size = params.get('target_vocab_size')
            self.embedding_dim = params.get('embedding_dim')
            assert self.embedding_dim is not None and self.source_vocab_size is not None and self.target_vocab_size is not None
        self.output_size = self.target_vocab_size
        self.num_layers = params.get('num_layers', 1)
        self.dropout = params.get('dropout', 0.5)
        self.embed_dropout = params.get('embed_dropout')
        self.train_embedding = params.get('train_embedding', False)

        # Initialize embeddings. Static embeddings for now.
        self.source_embeddings = t.nn.Embedding(self.source_vocab_size, self.embedding_dim)
        self.target_embeddings = t.nn.Embedding(self.target_vocab_size, self.embedding_dim)
        if source_embeddings is not None:
            self.source_embeddings.weight = t.nn.Parameter(source_embeddings, requires_grad=self.train_embedding)
        if target_embeddings is not None:
            self.target_embeddings.weight = t.nn.Parameter(target_embeddings, requires_grad=self.train_embedding)

        # Initialize network modules.
        self.encoder_rnn = t.nn.LSTM(self.embedding_dim, self.hidden_dim, dropout=self.dropout, num_layers=self.num_layers, batch_first=True)
        self.decoder_rnn = t.nn.LSTM(self.embedding_dim + self.hidden_dim, self.hidden_dim, dropout=self.dropout, num_layers=self.num_layers, batch_first=True)
        self.hidden2out = t.nn.Linear(self.hidden_dim, self.output_size)
        self.hidden_enc = self.init_hidden()
        self.hidden_dec = self.init_hidden()
        if self.embed_dropout:
            self.dropout_1s = t.nn.Dropout(self.dropout)
            self.dropout_1t = t.nn.Dropout(self.dropout)
        self.dropout_2 = t.nn.Dropout(self.dropout)

    def init_hidden(self, batch_size=None):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim). The helper function
        # will return torch variable.
        bs = self.batch_size if batch_size is None else batch_size
        return tuple((
            variable(np.zeros((self.num_layers, bs, self.hidden_dim)), cuda=self.cuda_flag),
            variable(np.zeros((self.num_layers, bs, self.hidden_dim)), cuda=self.cuda_flag)
        ))

    def forward(self, x_source, x_target):
        """
        :param x_source: the source sentence
        :param x_target: the target (translated) sentence
        :return:
        """
        # EMBEDDING
        xx_source = self.reverse_source(x_source)
        embedded_x_source = self.source_embeddings(xx_source)
        embedded_x_target = self.source_embeddings(x_target[:, :-1])
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
        return out_linear

    def translate(self, x_source):
        # INITIALIZE
        self.eval()

        self.hidden_enc = self.init_hidden()
        self.hidden_dec = self.init_hidden()
        hidden = self.hidden_dec

        count_eos = 0
        time = 0

        x_target = (2 * t.ones(x_source.size(0), 1)).long()  # `2` is the SOS token (<s>)
        x_target = variable(x_target, to_float=False, cuda=self.cuda_flag)

        # EMBEDDING
        xx_source = self.reverse_source(x_source)
        embedded_x_source = self.source_embeddings(xx_source)
        if self.embed_dropout:
            embedded_x_source = self.dropout_1s(embedded_x_source)

        # ENCODING SOURCE SENTENCE INTO FIXED LENGTH VECTOR
        _, self.hidden_enc = self.encoder_rnn(embedded_x_source, self.hidden_enc)

        while count_eos < x_source.size(0):
            embedded_x_target = self.target_embeddings(x_target)
            embedded_x_target = self.append_hidden_to_target(embedded_x_target)
            dec_out, hidden = self.decoder_rnn(embedded_x_target, hidden)
            hidden = hidden[0].detach(), hidden[1].detach()
            dec_out = dec_out[:, time:time + 1, :].detach()
            dec_out = self.dropout_2(dec_out)

            # OUTPUT
            pred = self.hidden2out(dec_out).detach()
            # concatenate the output of the decoder and the context and apply nonlinearity
            x_target = t.cat([x_target, pred.max(2)[1]], 1).detach()

            # should you stop ?
            count_eos += t.sum((pred.max(2)[1] == 3).long()).data.cpu().numpy()[0]  # `3` is the EOS token
            time += 1
        return x_target

    @staticmethod
    def reverse_source(x):
        """
        Reverse the source sentence x. Empirically it was observed to work better in terms of final valid PPL, and especially for long sentences
        `x` is the integer-encoded sentence. It is a batch x sentence_length LongTensor
        """
        # asssume that the batch_size is the first dim
        return variable(t.cat([x.data[:, -1:]] + [x.data[:, -(k + 1):-k] for k in range(1, x.size(1))], 1), to_float=False)

    def append_hidden_to_target(self, x):
        """Append self.hidden_enc to all timesteps of x"""
        # self.hidden_enc[0] this is h. Size num_layers x batch x hdim
        h = self.hidden_enc[0]
        # h[-1:, :, :].permute(1,0,2) this is h for the last layer. Size batch x 1 x hdim
        h_last = h[-1:, :, :].permute(1, 0, 2)
        hidden = t.cat(x.size(1) * [h_last], 1)
        return t.cat([x, hidden], 2)


class LSTMA(t.nn.Module):
    """
    Implementation of `Neural Machine Translation by Jointly Learning to Align and Translate`
    https://arxiv.org/abs/1409.0473

    NOTE THAT ITS INPUT SHOULD HAVE THE BATCH SIZE FIRST !!!!!
    """

    def __init__(self, params, source_embeddings=None, target_embeddings=None):
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
            self.source_vocab_size = params.get('source_vocab_size')
            self.target_vocab_size = params.get('target_vocab_size')
        except:
            # if you dont provide a pre-trained embedding, you have to provide these
            self.embedding_dim = params.get('embedding_dim')
            self.source_vocab_size = params.get('source_vocab_size')
            self.target_vocab_size = params.get('target_vocab_size')
            assert self.embedding_dim is not None and self.source_vocab_size is not None and self.target_vocab_size is not None
        self.output_size = self.target_vocab_size
        self.num_layers = params.get('num_layers', 1)
        self.dropout = params.get('dropout', 0.5)
        self.embed_dropout = params.get('embed_dropout')
        self.train_embedding = params.get('train_embedding', True)

        # Initialize embeddings. Static embeddings for now.
        self.source_embeddings = t.nn.Embedding(self.source_vocab_size, self.embedding_dim)
        self.target_embeddings = t.nn.Embedding(self.target_vocab_size, self.embedding_dim)
        if source_embeddings is not None:
            self.source_embeddings.weight = t.nn.Parameter(source_embeddings, requires_grad=self.train_embedding)
        if target_embeddings is not None:
            self.target_embeddings.weight = t.nn.Parameter(target_embeddings, requires_grad=self.train_embedding)

        # Initialize network modules.
        # note that the encoder is a BiLSTM. The output is modified by the fact that the hidden dim is doubled, and if you set
        # the number of layers to L, there will actually be 2L layers (the forward ones and the backward ones). Consequently the first
        # dimension of the hidden outputs of the forward pass (the 2nd output in the tuple) will be a tuple of
        # 2 tensors having as first dim twice the hidden dim you set
        self.encoder_rnn = t.nn.LSTM(self.embedding_dim, self.hidden_dim // 2, dropout=self.dropout, num_layers=self.num_layers, bidirectional=True, batch_first=True)
        self.decoder_rnn = t.nn.LSTM(self.embedding_dim, self.hidden_dim, dropout=self.dropout, num_layers=self.num_layers, batch_first=True)
        self.hidden_dec_initializer = t.nn.Linear(self.hidden_dim // 2, self.hidden_dim)
        self.hidden2out = t.nn.Linear(self.hidden_dim * 2, self.output_size)
        if self.embed_dropout:
            self.dropout_1s = t.nn.Dropout(self.dropout)
            self.dropout_1t = t.nn.Dropout(self.dropout)
        self.dropout_2 = t.nn.Dropout(self.dropout)

        if self.cuda_flag:
            self = self.cuda()

    # @todo: maybe this is wrong in case of deep LSTM DECODER (I am not sure the dimensions are correct)
    def init_hidden(self, data, type):
        """
        Initialize the hidden state, either for the encoder or the decoder

        For type=`enc`, it should just be initialized with 0s
        For type=`dec`, it should be initialized with tanh(W h1_backward) (see page 13 of the paper, last paragraph)

        `data` is either something you initialize the hidden state with, or None
        """
        if type == 'dec':
            # in that case, `data` is the output of the encoder
            # data[:, :1, self.hidden_dim // 2:]
            # `:` for the whole batch
            # `:1` because you want the hidden state of the first time step (see paper, they use backward(h1))
            # but also `self.hidden_dim // 2:`, because you want the backward part only (the last coefficients)
            h = F.tanh(self.hidden_dec_initializer(data[:, :1, self.hidden_dim // 2:]))  # @todo: verify that the last hdim/2 weights actually correspond to the backward layer(s)
            h = h.transpose(1, 0)
            return (
                h,
                variable(np.zeros((self.num_layers, self.batch_size, self.hidden_dim)), cuda=self.cuda_flag)
            )
        elif type == 'enc':
            # in that case data is None
            return tuple((
                variable(np.zeros((self.num_layers * 2, self.batch_size, self.hidden_dim // 2)), cuda=self.cuda_flag),
                variable(np.zeros((self.num_layers * 2, self.batch_size, self.hidden_dim // 2)), cuda=self.cuda_flag)
            ))
        else:
            raise ValueError('the type should be either `dec` or `enc`')

    def forward(self, x_source, x_target):
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
        attn_dist = F.softmax(scores, dim=1)  # batch x source_len x target_len
        context = t.bmm(attn_dist.permute(0, 2, 1), enc_out)  # batch x target_len x hidden_dim

        # OUTPUT
        # concatenate the output of the decoder and the context and apply nonlinearity
        pred = F.tanh(t.cat([dec_out, context], -1))  # @todo : tanh necessary ?
        pred = self.dropout_2(pred)  # batch x target_len x 2 hdim
        pred = self.hidden2out(pred)
        return pred

    def translate(self, x_source):
        self.eval()

        # EMBEDDING
        embedded_x_source = self.source_embeddings(x_source)
        if self.embed_dropout:
            embedded_x_source = self.dropout_1s(embedded_x_source)

        # RECURRENT
        hidden = self.init_hidden(None, 'enc')
        enc_out, _ = self.encoder_rnn(embedded_x_source, hidden)
        hidden = self.init_hidden(enc_out, 'dec')
        x_target = (2 * t.ones(x_source.size(0), 1)).long()  # `2` is the SOS token (<s>)
        x_target = variable(x_target, to_float=False, cuda=self.cuda_flag)
        count_eos = 0
        time = 0
        while count_eos < x_source.size(0):
            embedded_x_target = self.target_embeddings(x_target)
            dec_out, hidden = self.decoder_rnn(embedded_x_target, hidden)
            hidden = hidden[0].detach(), hidden[1].detach()
            dec_out = dec_out[:, time:time + 1, :].detach()

            # ATTENTION
            scores = t.bmm(enc_out, dec_out.transpose(1, 2))  # this will be a batch x source_len x target_len
            try:
                attn_dist = F.softmax(scores, dim=1)  # batch x source_len x target_len
            except:
                attn_dist = F.softmax(scores.permute(1, 0, 2)).permute(1, 0, 2)
            context = t.bmm(attn_dist.permute(0, 2, 1), enc_out)  # batch x target_len x hidden_dim

            # OUTPUT
            # concatenate the output of the decoder and the context and apply nonlinearity
            pred = F.tanh(t.cat([dec_out, context], -1))  # @todo : tanh necessary ?
            pred = self.dropout_2(pred)  # batch x target_len x 2 hdim
            pred = self.hidden2out(pred).detach()
            x_target = t.cat([x_target, pred.max(2)[1]], 1).detach()

            # should you stop ?
            count_eos += t.sum((pred.max(2)[1] == 3).long()).data.cpu().numpy()[0]  # `3` is the EOS token
            time += 1
        return x_target


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
