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

os.chdir('../HW3')  # so that there is not any import bug in case HW3 is not already the working directory
from utils import *
from const import SOS_TOKEN, PAD_TOKEN, EOS_TOKEN, CUDA_DEFAULT
from copy import deepcopy


class LSTM(t.nn.Module):
    """
    Implementation of `Sequence to Sequence Learning with Neural Networks`
    https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf

    Here the input sentence is not reversed. See LSTMR for this

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
        self.decoder_rnn = t.nn.LSTM(self.embedding_dim, self.hidden_dim, dropout=self.dropout, num_layers=self.num_layers, batch_first=True)
        self.hidden2out = t.nn.Linear(self.hidden_dim * 2, self.output_size)
        self.hidden_enc = self.init_hidden('enc')
        self.hidden_dec = self.init_hidden('dec')
        if self.embed_dropout:
            self.dropout_1s = t.nn.Dropout(self.dropout)
            self.dropout_1t = t.nn.Dropout(self.dropout)
        self.dropout_2 = t.nn.Dropout(self.dropout)

    def init_hidden(self, type, batch_size=None):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim). The helper function
        # will return torch variable.
        bs = self.batch_size if batch_size is None else batch_size
        nl = self.num_layers
        return tuple((
            variable(np.zeros((nl, bs, self.hidden_dim)), cuda=self.cuda_flag),
            variable(np.zeros((nl, bs, self.hidden_dim)), cuda=self.cuda_flag)
        ))

    def forward(self, x_source, x_target):
        """
        :param x_source: the source sentence (batch x sentence_length)
        :param x_target: the target (translated) sentence (batch x sentence_length)
        :return:
        """
        # EMBEDDING
        embedded_x_source = self.source_embeddings(x_source)
        embedded_x_target = self.target_embeddings(x_target[:, :-1])  # don't take into account the last token because there is nothing after
        if self.embed_dropout:
            embedded_x_source = self.dropout_1s(embedded_x_source)
            embedded_x_target = self.dropout_1t(embedded_x_target)

        # ENCODING SOURCE SENTENCE INTO FIXED LENGTH VECTOR
        enc_output, self.hidden_enc = self.encoder_rnn(embedded_x_source, self.hidden_enc)
        context = enc_output[:, -1, :].unsqueeze(1)  # batch x hdim
        context = context.repeat(1, x_target.size(1) - 1, 1)  # batch x target_length x hdim

        # DECODING
        rnn_out, self.hidden_dec = self.decoder_rnn(embedded_x_target, self.hidden_dec)
        rnn_out = F.tanh(t.cat([rnn_out, context], -1))
        rnn_out = self.dropout_2(rnn_out)

        # OUTPUT
        out_linear = self.hidden2out(rnn_out)
        return out_linear

    def translate(self, x_source):
        # INITIALIZE
        self.eval()

        self.hidden_enc = self.init_hidden('enc', x_source.size(0))
        self.hidden_dec = self.init_hidden('dec', x_source.size(0))
        hidden = self.hidden_dec

        count_eos = 0
        time = 0

        x_target = (SOS_TOKEN * t.ones(x_source.size(0), 1)).long()  # `2` is the SOS token (<s>)
        x_target = variable(x_target, to_float=False, cuda=self.cuda_flag)

        # EMBEDDING
        embedded_x_source = self.source_embeddings(x_source)
        if self.embed_dropout:
            embedded_x_source = self.dropout_1s(embedded_x_source)

        # ENCODING SOURCE SENTENCE INTO FIXED LENGTH VECTOR
        _, self.hidden_enc = self.encoder_rnn(embedded_x_source, self.hidden_enc)
        context = _[:, -1, :].unsqueeze(1)  # batch x hdim

        while count_eos < x_source.size(0):
            embedded_x_target = self.target_embeddings(x_target)
            dec_out, hidden = self.decoder_rnn(embedded_x_target, hidden)
            hidden = hidden[0].detach(), hidden[1].detach()
            dec_out = dec_out[:, time:time + 1, :].detach()
            dec_out = F.tanh(t.cat([dec_out, context], -1))
            dec_out = self.dropout_2(dec_out)

            # OUTPUT
            pred = self.hidden2out(dec_out).detach()
            # concatenate the output of the decoder and the context and apply nonlinearity
            x_target = t.cat([x_target, pred.max(2)[1]], 1).detach()

            # should you stop ?
            count_eos += t.sum((pred.max(2)[1] == EOS_TOKEN).long()).data.cpu().numpy()[0]  # `3` is the EOS token
            time += 1
        return x_target


class LSTMR(t.nn.Module):
    """
    Implementation of `Sequence to Sequence Learning with Neural Networks`
    https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf

    NOTE THAT ITS INPUT SHOULD HAVE THE BATCH SIZE FIRST !!!!!
    """

    def __init__(self, params, source_embeddings=None, target_embeddings=None):
        super(LSTMR, self).__init__()
        print("Initializing LSTMR")
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
        self.blstm_enc = params.get('blstm_enc', False)

        # Initialize embeddings. Static embeddings for now.
        self.source_embeddings = t.nn.Embedding(self.source_vocab_size, self.embedding_dim)
        self.target_embeddings = t.nn.Embedding(self.target_vocab_size, self.embedding_dim)
        if source_embeddings is not None:
            self.source_embeddings.weight = t.nn.Parameter(source_embeddings, requires_grad=self.train_embedding)
        if target_embeddings is not None:
            self.target_embeddings.weight = t.nn.Parameter(target_embeddings, requires_grad=self.train_embedding)

        # Initialize network modules.
        self.encoder_rnn = t.nn.LSTM(self.embedding_dim,
                                     (self.hidden_dim // 2) * self.blstm_enc + self.hidden_dim * (1 - self.blstm_enc),
                                     dropout=self.dropout, num_layers=self.num_layers, batch_first=True, bidirectional=self.blstm_enc)
        self.decoder_rnn = t.nn.LSTM(self.embedding_dim, self.hidden_dim, dropout=self.dropout, num_layers=self.num_layers, batch_first=True)
        self.hidden2out = t.nn.Linear(self.hidden_dim * 2, self.output_size)
        self.hidden_enc = self.init_hidden('enc')
        self.hidden_dec = self.init_hidden('dec')
        if self.embed_dropout:
            self.dropout_1s = t.nn.Dropout(self.dropout)
            self.dropout_1t = t.nn.Dropout(self.dropout)
        self.dropout_2 = t.nn.Dropout(self.dropout)

    def init_hidden(self, type, batch_size=None):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim). The helper function
        # will return torch variable.
        bs = self.batch_size if batch_size is None else batch_size
        nl = self.num_layers
        if type == 'enc' and self.blstm_enc:
            nl *= 2
        return tuple((
            variable(np.zeros((nl, bs, self.hidden_dim)), cuda=self.cuda_flag),
            variable(np.zeros((nl, bs, self.hidden_dim)), cuda=self.cuda_flag)
        ))

    def forward(self, x_source, x_target):
        """
        :param x_source: the source sentence (batch x sentence_length)
        :param x_target: the target (translated) sentence (batch x sentence_length)
        :return:
        """
        # EMBEDDING
        if not self.blstm_enc:
            xx_source = self.reverse_source(x_source)
        else:
            xx_source = x_source
        embedded_x_source = self.source_embeddings(xx_source)
        embedded_x_target = self.target_embeddings(x_target[:, :-1])  # don't take into account the last token because there is nothing after
        if self.embed_dropout:
            embedded_x_source = self.dropout_1s(embedded_x_source)
            embedded_x_target = self.dropout_1t(embedded_x_target)

        # ENCODING SOURCE SENTENCE INTO FIXED LENGTH VECTOR
        _, self.hidden_enc = self.encoder_rnn(embedded_x_source, self.hidden_enc)
        context = _[:, -1, :].unsqueeze(1)  # batch x hdim
        context = context.repeat(1, x_target.size(1) - 1, 1)  # batch x target_length x hdim

        # DECODING
        rnn_out, self.hidden_dec = self.decoder_rnn(embedded_x_target, self.hidden_dec)
        rnn_out = F.tanh(t.cat([rnn_out, context], -1))
        rnn_out = self.dropout_2(rnn_out)

        # OUTPUT
        out_linear = self.hidden2out(rnn_out)
        return out_linear

    def translate(self, x_source):
        # INITIALIZE
        self.eval()

        self.hidden_enc = self.init_hidden('enc')
        self.hidden_dec = self.init_hidden('dec')
        hidden = self.hidden_dec

        count_eos = 0
        time = 0

        x_target = (SOS_TOKEN * t.ones(x_source.size(0), 1)).long()  # `2` is the SOS token (<s>)
        x_target = variable(x_target, to_float=False, cuda=self.cuda_flag)

        # EMBEDDING
        if not self.blstm_enc:
            xx_source = self.reverse_source(x_source)
        else:
            xx_source = x_source
        embedded_x_source = self.source_embeddings(xx_source)
        if self.embed_dropout:
            embedded_x_source = self.dropout_1s(embedded_x_source)

        # ENCODING SOURCE SENTENCE INTO FIXED LENGTH VECTOR
        _, self.hidden_enc = self.encoder_rnn(embedded_x_source, self.hidden_enc)
        context = _[:, -1, :].unsqueeze(1)  # batch x hdim

        while count_eos < x_source.size(0):
            embedded_x_target = self.target_embeddings(x_target)
            embedded_x_target = self.append_hidden_to_target(embedded_x_target)
            dec_out, hidden = self.decoder_rnn(embedded_x_target, hidden)
            hidden = hidden[0].detach(), hidden[1].detach()
            dec_out = dec_out[:, time:time + 1, :].detach()
            dec_out = F.tanh(t.cat([dec_out, context], -1))
            dec_out = self.dropout_2(dec_out)

            # OUTPUT
            pred = self.hidden2out(dec_out).detach()
            # concatenate the output of the decoder and the context and apply nonlinearity
            x_target = t.cat([x_target, pred.max(2)[1]], 1).detach()

            # should you stop ?
            count_eos += t.sum((pred.max(2)[1] == EOS_TOKEN).long()).data.cpu().numpy()[0]  # `3` is the EOS token
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
        self.weight_norm = params.get('wn', False)

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
        if self.weight_norm:
            self.wn()
        self.hidden_dec_initializer = t.nn.Linear(self.hidden_dim // 2, self.num_layers * self.hidden_dim)
        self.hidden2out = t.nn.Linear(self.hidden_dim * 2, self.output_size)
        if self.embed_dropout:
            self.dropout_1s = t.nn.Dropout(self.dropout)
            self.dropout_1t = t.nn.Dropout(self.dropout)
        self.dropout_2 = t.nn.Dropout(self.dropout)
        self.lsm = nn.LogSoftmax()

        self.beam_size = params.get('beam_size', 3)
        self.max_beam_depth = params.get('max_beam_depth', 20)

        if self.cuda_flag:
            self = self.cuda()

    def init_hidden(self, data, type, batch_size=None):
        """
        Initialize the hidden state, either for the encoder or the decoder

        For type=`enc`, it should just be initialized with 0s
        For type=`dec`, it should be initialized with tanh(W h1_backward) (see page 13 of the paper, last paragraph)

        `data` is either something you initialize the hidden state with, or None
        """
        bs = batch_size if batch_size is not None else self.batch_size
        if type == 'dec':
            # in that case, `data` is the output of the encoder
            # data[:, :1, self.hidden_dim // 2:]
            # `:` for the whole batch
            # `:1` because you want the hidden state of the first time step (see paper, they use backward(h1))
            # but also `self.hidden_dim // 2:`, because you want the backward part only (the last coefficients)
            h = F.tanh(self.hidden_dec_initializer(data[:, :1, self.hidden_dim // 2:]))  # the last hdim/2 weights correspond to the backward layer(s)
            h = h.transpose(1, 0)
            h = t.cat(t.split(h, self.hidden_dim, dim=2), 0)
            return (
                h,
                variable(np.zeros((self.num_layers, bs, self.hidden_dim)), cuda=self.cuda_flag)
            )
        elif type == 'enc':
            # in that case data is None
            return tuple((
                variable(np.zeros((self.num_layers * 2, bs, self.hidden_dim // 2)), cuda=self.cuda_flag),
                variable(np.zeros((self.num_layers * 2, bs, self.hidden_dim // 2)), cuda=self.cuda_flag)
            ))
        else:
            raise ValueError('the type should be either `dec` or `enc`')

    def wn(self):
        for i in range(self.num_layers):
            self.encoder_rnn = t.nn.utils.weight_norm(self.encoder_rnn, 'weight_hh_l%d' % i)
            self.encoder_rnn = t.nn.utils.weight_norm(self.encoder_rnn, 'weight_hh_l%d_reverse' % i)
            self.encoder_rnn = t.nn.utils.weight_norm(self.encoder_rnn, 'weight_ih_l%d' % i)
            self.encoder_rnn = t.nn.utils.weight_norm(self.encoder_rnn, 'weight_ih_l%d_reverse' % i)
            self.decoder_rnn = t.nn.utils.weight_norm(self.decoder_rnn, 'weight_hh_l%d' % i)
            self.decoder_rnn = t.nn.utils.weight_norm(self.decoder_rnn, 'weight_ih_l%d' % i)

    def forward(self, x_source, x_target, return_attn=False):
        # EMBEDDING
        embedded_x_source = self.source_embeddings(x_source)
        embedded_x_target = self.target_embeddings(x_target[:, :-1])  # don't make a prediction for the word following the last one
        if self.embed_dropout:
            embedded_x_source = self.dropout_1s(embedded_x_source)
            embedded_x_target = self.dropout_1t(embedded_x_target)

        # RECURRENT
        hidden = self.init_hidden(None, 'enc', x_source.size(0))
        enc_out, _ = self.encoder_rnn(embedded_x_source, hidden)
        hidden = self.init_hidden(enc_out, 'dec', x_source.size(0))
        dec_out, _ = self.decoder_rnn(embedded_x_target, hidden)

        # ATTENTION
        scores = t.bmm(enc_out, dec_out.transpose(1, 2))  # this will be a batch x source_len x target_len
        attn_dist = F.softmax(scores, dim=1)  # batch x source_len x target_len
        context = t.bmm(attn_dist.permute(0, 2, 1), enc_out)  # batch x target_len x hidden_dim

        # OUTPUT
        # concatenate the output of the decoder and the context and apply nonlinearity
        pred = F.tanh(t.cat([dec_out, context], -1))
        pred = self.dropout_2(pred)  # batch x target_len x 2 hdim
        pred = self.hidden2out(pred)

        if return_attn:
            return pred, attn_dist
        else:
            return pred

    def translate(self, x_source):
        self.eval()

        # EMBEDDING
        embedded_x_source = self.source_embeddings(x_source)
        if self.embed_dropout:
            embedded_x_source = self.dropout_1s(embedded_x_source)

        # RECURRENT
        hidden = self.init_hidden(None, 'enc', x_source.size(0))
        enc_out, _ = self.encoder_rnn(embedded_x_source, hidden)
        hidden = self.init_hidden(enc_out, 'dec', x_source.size(0))
        x_target = (SOS_TOKEN * t.ones(x_source.size(0), 1)).long()  # `2` is the SOS token (<s>)
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
            pred = F.tanh(t.cat([dec_out, context], -1))
            pred = self.dropout_2(pred)  # batch x target_len x 2 hdim
            pred = self.hidden2out(pred).detach()
            x_target = t.cat([x_target, pred.max(2)[1]], 1).detach()

            # should you stop ?
            count_eos += t.sum((pred.max(2)[1] == EOS_TOKEN).long()).data.cpu().numpy()[0]  # `3` is the EOS token
            time += 1
        return x_target

    def translate_beam(self, x_source, print_beam_row=-1):
        self.eval()

        # EMBEDDING
        embedded_x_source = self.source_embeddings(x_source)
        if self.embed_dropout:
            embedded_x_source = self.dropout_1s(embedded_x_source)

        terminate_beam = False
        batch_size = x_source.size(0)

        # RECURRENT
        hidden = self.init_hidden(None, 'enc', x_source.size(0))
        enc_out, _ = self.encoder_rnn(embedded_x_source, hidden)

        # One hidden for each beam element.
        hidden = []
        for i in range(self.beam_size):
            hidden.append(self.init_hidden(enc_out, 'dec', x_source.size(0)))

        x_target = SOS_TOKEN * np.ones((x_source.size(0), 1))  # `2` is the SOS token (<s>)
        count_eos = 0
        time = 0

        # INIT SOME STUFF.
        self.beam = np.array([x_target])
        self.beam_scores = np.zeros((batch_size, 1))

        while not terminate_beam and time < self.max_beam_depth:
            collective_children = np.array([])
            collective_scores = np.array([])

            if len(self.beam) == 1:
                reshaped_beam = self.beam
            else:
                reshaped_beam = np.transpose(self.beam, (1, 0, 2))

            for it, elem in enumerate(reshaped_beam):
                elem = t.from_numpy(elem).long()
                x_target = elem.contiguous().view(self.batch_size, -1)
                x_target = variable(x_target, to_float=False, cuda=self.cuda_flag).long()
                embedded_x_target = self.target_embeddings(x_target)
                dec_out, hidden_out = self.decoder_rnn(embedded_x_target, hidden[it])
                hidden[it] = hidden_out[0].detach(), hidden_out[1].detach()
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
                pred = F.tanh(t.cat([dec_out, context], -1))
                pred = self.dropout_2(pred)  # batch x target_len x 2 hdim
                pred = self.hidden2out(pred).detach()
                pred = self.lsm(pred.view(batch_size, -1)).detach()

                topk = t.topk(pred, self.beam_size, dim=1)
                top_k_indices, top_k_scores = topk[1], topk[0]
                top_k_indices = top_k_indices.transpose(0, 1)
                top_k_scores = top_k_scores.transpose(0, 1)

                for new_word_batch, new_score_batch in zip(top_k_indices, top_k_scores):
                    new_word_batch = new_word_batch.contiguous().view(batch_size, 1)
                    new_score_batch = new_score_batch.contiguous().view(batch_size, 1)

                    new_child_batch = t.cat([x_target, new_word_batch], 1).detach()

                    batch_parent_score = self.beam_scores[:, it].reshape((self.batch_size, 1))
                    batch_acc_score = batch_parent_score + new_score_batch.data.cpu().numpy()

                    if len(collective_children) > 0:
                        collective_children = np.hstack((collective_children, new_child_batch.data.cpu().numpy()))
                        # Add the corresponding beam element's score with the new score and stack it.
                        collective_scores = np.hstack((collective_scores, batch_acc_score))
                    else:
                        collective_children, collective_scores = new_child_batch.data.cpu().numpy(), batch_acc_score


                        # At the end of a for loop collective children, collective scores
            # will look a numpy array of tensors.
            current_beam_length = 1  # Means only start elem is there.
            if len(self.beam) != 1:
                current_beam_length = self.beam.shape[1]

            collective_children = collective_children.reshape((batch_size, current_beam_length * self.beam_size,
                                                               int(collective_children.shape[1] /
                                                                   current_beam_length / self.beam_size)
                                                               ))

            if collective_children.shape[1] == self.beam_size:  # Happens the first time.
                self.beam = collective_children
                self.beam_scores = collective_scores
            else:
                self.beam = deepcopy(np.zeros((batch_size, self.beam_size, collective_children.shape[2])))
                for i in range(batch_size):
                    # Since argsort gives ascending order
                    best_scores_indices = np.argsort(-1 * collective_scores[i])[:self.beam_size]
                    for key, index in enumerate(best_scores_indices):
                        self.beam[i][key][:] = collective_children[i][index]
                        self.beam_scores[i][key] = collective_scores[i][index]

            terminate_beam = True

            for x in self.beam:
                for c in x:
                    if EOS_TOKEN not in c:
                        terminate_beam = False
                        break
                if not terminate_beam:
                    break
                    # import pdb; pdb.set_trace()
            assert (self.beam.shape == (batch_size, self.beam_size, time + 2))

            time += 1
            print(time)
        return self.beam


class LSTMF(t.nn.Module):
    """
    Implementation of `Neural Machine Translation by Jointly Learning to Align and Translate`
    https://arxiv.org/abs/1409.0473

    NOTE THAT ITS INPUT SHOULD HAVE THE BATCH SIZE FIRST !!!!!
    """

    def __init__(self, params, source_embeddings=None, target_embeddings=None):
        super(LSTMF, self).__init__()
        print("Initializing LSTMF")
        self.cuda_flag = params.get('cuda', CUDA_DEFAULT)
        self.model_str = 'LSTMF'
        self.params = params

        # Initialize hyperparams.
        self.hidden_dim = params.get('hidden_dim', 100)
        self.batch_size = params.get('batch_size', 32)
        self.dropout = params.get('dropout', 0.5)
        self.embed_dropout = params.get('embed_dropout')
        self.initialize_embeddings(params, source_embeddings, target_embeddings)
        self.output_size = self.target_vocab_size
        assert self.hidden_dim == self.embedding_dim

        # Initialize network modules.
        self.encoder_rnn1 = t.nn.LSTM(self.embedding_dim, self.hidden_dim // 2, dropout=self.dropout, num_layers=1, bidirectional=True, batch_first=True)
        self.encoder_rnn2 = t.nn.LSTM(self.embedding_dim, self.hidden_dim, dropout=self.dropout, num_layers=1, bidirectional=False, batch_first=True)
        self.decoder_rnn1 = t.nn.LSTM(self.embedding_dim, self.hidden_dim, dropout=self.dropout, num_layers=1, batch_first=True)
        self.decoder_rnn2 = t.nn.LSTM(self.embedding_dim, self.hidden_dim, dropout=self.dropout, num_layers=1, batch_first=True)
        self.hidden_dec_initializer = t.nn.Linear(self.hidden_dim // 2, 2 * self.hidden_dim)
        self.hidden2out = t.nn.Linear(self.hidden_dim * 2, self.output_size)
        if self.embed_dropout:
            self.dropout_1s = t.nn.Dropout(self.dropout)
            self.dropout_1t = t.nn.Dropout(self.dropout)
        self.dropout_1_enc = t.nn.Dropout(self.dropout)
        self.dropout_1_dec = t.nn.Dropout(self.dropout)
        self.dropout_2 = t.nn.Dropout(self.dropout)
        self.linear_enc = t.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_dec = t.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_attn = t.nn.Linear(self.hidden_dim, 1, bias=False)
        self.lsm = nn.LogSoftmax()

        self.beam_size = params.get('beam_size', 3)
        self.max_beam_depth = params.get('max_beam_depth', 20)

        if self.cuda_flag:
            self = self.cuda()

    def initialize_embeddings(self, params, source_embeddings, target_embeddings):
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

        self.train_embedding = params.get('train_embedding', True)

        # Initialize embeddings. Static embeddings for now.
        self.source_embeddings = t.nn.Embedding(self.source_vocab_size, self.embedding_dim)
        self.target_embeddings = t.nn.Embedding(self.target_vocab_size, self.embedding_dim)
        if source_embeddings is not None:
            self.source_embeddings.weight = t.nn.Parameter(source_embeddings, requires_grad=self.train_embedding)
        if target_embeddings is not None:
            self.target_embeddings.weight = t.nn.Parameter(target_embeddings, requires_grad=self.train_embedding)

    def init_hidden(self, data, type, batch_size=None):
        """
        Initialize the hidden state, either for the encoder or the decoder

        For type=`enc`, it should just be initialized with 0s
        For type=`dec`, it should be initialized with tanh(W h1_backward) (see page 13 of the paper, last paragraph)

        `data` is either something you initialize the hidden state with, or None
        """
        bs = batch_size if batch_size is not None else self.batch_size
        if type == 'dec':
            # in that case, `data` is the output of the encoder
            # data[:, :1, self.hidden_dim // 2:]
            # `:` for the whole batch
            # `:1` because you want the hidden state of the first time step (see paper, they use backward(h1))
            # but also `self.hidden_dim // 2:`, because you want the backward part only (the last coefficients)
            h = F.tanh(self.hidden_dec_initializer(data[:, -1:, self.hidden_dim // 2:]))  # the last hdim/2 weights correspond to the backward layer(s)
            h = h.transpose(1, 0)
            h = t.cat(t.split(h, self.hidden_dim, dim=2), 0)
            return (
                h,
                variable(np.zeros((2, bs, self.hidden_dim)), cuda=self.cuda_flag)
            )
        elif type == 'enc1':
            # in that case data is None
            return tuple((
                variable(np.zeros((2, bs, self.hidden_dim // 2)), cuda=self.cuda_flag),
                variable(np.zeros((2, bs, self.hidden_dim // 2)), cuda=self.cuda_flag)
            ))
        elif type == 'enc2':
            # in that case data is None
            return tuple((
                variable(np.zeros((1, bs, self.hidden_dim)), cuda=self.cuda_flag),
                variable(np.zeros((1, bs, self.hidden_dim)), cuda=self.cuda_flag)
            ))
        else:
            raise ValueError('the type should be either `dec` or `enc`')

    def forward(self, x_source, x_target, return_attn=False):
        batch_size = x_source.size(0)
        src_len = x_source.size(1)
        trg_len = x_target.size(1) - 1

        # EMBEDDING
        embedded_x_source = self.source_embeddings(x_source)
        embedded_x_target = self.target_embeddings(x_target[:, :-1])  # don't make a prediction for the word following the last one
        if self.embed_dropout:
            embedded_x_source = self.dropout_1s(embedded_x_source)
            embedded_x_target = self.dropout_1t(embedded_x_target)

        # RECURRENT: 2 layers of encoder and 2 layers of decoder. There is dropout inbetween layers, and skip connections as well
        # encoder
        hidden1 = self.init_hidden(None, 'enc1', batch_size)
        hidden2 = self.init_hidden(None, 'enc2', batch_size)
        enc_out, _ = self.encoder_rnn1(embedded_x_source, hidden1)
        enc_out, _ = self.encoder_rnn2(self.dropout_1_enc(embedded_x_source + enc_out), hidden2)  # skip connection + dropout
        # decoder
        hidden12 = self.init_hidden(enc_out, 'dec', batch_size)
        hidden1 = hidden12[0][:1, :, :], hidden12[1][:1, :, :]
        hidden2 = hidden12[0][1:, :, :], hidden12[1][1:, :, :]
        dec_out, _ = self.decoder_rnn1(embedded_x_target, hidden1)
        dec_out, _ = self.decoder_rnn2(self.dropout_1_dec(embedded_x_target + dec_out), hidden2)

        # ATTENTION: just like in the paper
        scores = self.linear_attn(F.tanh(
            self.linear_enc(enc_out).unsqueeze(2).expand(batch_size, src_len, trg_len, self.hidden_dim) +
            self.linear_dec(dec_out).unsqueeze(1).expand(batch_size, src_len, trg_len, self.hidden_dim)
        )).squeeze(3)
        attn_dist = F.softmax(scores, dim=1)  # batch x source_len x target_len
        context = t.bmm(attn_dist.permute(0, 2, 1), enc_out)  # batch x target_len x hidden_dim

        # OUTPUT
        # concatenate the output of the decoder and the context and apply nonlinearity
        pred = F.tanh(t.cat([dec_out, context], -1))
        pred = self.dropout_2(pred)  # batch x target_len x 2 hdim
        pred = self.hidden2out(pred)

        if return_attn:
            return pred, attn_dist
        else:
            return pred

    def translate(self, x_source):
        batch_size = x_source.size(0)
        src_len = x_source.size(1)
        self.eval()

        # EMBEDDING
        embedded_x_source = self.source_embeddings(x_source)
        if self.embed_dropout:
            embedded_x_source = self.dropout_1s(embedded_x_source)

        # RECURRENT
        hidden1 = self.init_hidden(None, 'enc1', batch_size)
        hidden2 = self.init_hidden(None, 'enc2', batch_size)
        enc_out, _ = self.encoder_rnn1(embedded_x_source, hidden1)
        enc_out, _ = self.encoder_rnn2(self.dropout_1_enc(embedded_x_source + enc_out), hidden2)  # skip connection + dropout
        x_target = (SOS_TOKEN * t.ones(x_source.size(0), 1)).long()  # `2` is the SOS token (<s>)
        x_target = variable(x_target, to_float=False, cuda=self.cuda_flag)
        count_eos = 0
        time = 0
        while count_eos < x_source.size(0):
            embedded_x_target = self.target_embeddings(x_target)
            hidden12 = self.init_hidden(enc_out, 'dec', batch_size)
            hidden1 = hidden12[0][:1, :, :], hidden12[1][:1, :, :]
            hidden2 = hidden12[0][1:, :, :], hidden12[1][1:, :, :]
            dec_out, _ = self.decoder_rnn1(embedded_x_target, hidden1)
            dec_out, _ = self.decoder_rnn2(self.dropout_1_dec(embedded_x_target + dec_out), hidden2)
            dec_out = dec_out[:, time:time + 1, :].detach()

            # ATTENTION: just like in the paper
            scores = self.linear_attn(F.tanh(
                self.linear_enc(enc_out).unsqueeze(2).expand(batch_size, src_len, 1, self.hidden_dim) +
                self.linear_dec(dec_out).unsqueeze(1).expand(batch_size, src_len, 1, self.hidden_dim)
            )).squeeze(3)
            attn_dist = F.softmax(scores, dim=1)  # batch x source_len x target_len
            context = t.bmm(attn_dist.permute(0, 2, 1), enc_out)  # batch x target_len x hidden_dim

            # OUTPUT
            # concatenate the output of the decoder and the context and apply nonlinearity
            pred = F.tanh(t.cat([dec_out, context], -1))
            pred = self.dropout_2(pred)  # batch x target_len x 2 hdim
            pred = self.hidden2out(pred).detach()
            x_target = t.cat([x_target, pred.max(2)[1]], 1).detach()

            # should you stop ?
            count_eos += t.sum((pred.max(2)[1] == EOS_TOKEN).long()).data.cpu().numpy()[0]  # `3` is the EOS token
            time += 1
        return x_target

    # @todo: implement this
    def translate_beam(self, x_source, print_beam_row=-1):
        self.eval()

        # EMBEDDING
        embedded_x_source = self.source_embeddings(x_source)
        if self.embed_dropout:
            embedded_x_source = self.dropout_1s(embedded_x_source)

        terminate_beam = False
        batch_size = x_source.size(0)

        # RECURRENT
        hidden = self.init_hidden(None, 'enc', x_source.size(0))
        enc_out, _ = self.encoder_rnn(embedded_x_source, hidden)

        # One hidden for each beam element.
        hidden = []
        for i in range(self.beam_size):
            hidden.append(self.init_hidden(enc_out, 'dec', x_source.size(0)))

        x_target = SOS_TOKEN * np.ones((x_source.size(0), 1))  # `2` is the SOS token (<s>)
        count_eos = 0
        time = 0

        # INIT SOME STUFF.
        self.beam = np.array([x_target])
        self.beam_scores = np.zeros((batch_size, 1))

        while not terminate_beam and time < self.max_beam_depth:
            collective_children = np.array([])
            collective_scores = np.array([])

            if len(self.beam) == 1:
                reshaped_beam = self.beam
            else:
                reshaped_beam = np.transpose(self.beam, (1, 0, 2))

            for it, elem in enumerate(reshaped_beam):
                elem = t.from_numpy(elem).long()
                x_target = elem.contiguous().view(self.batch_size, -1)
                x_target = variable(x_target, to_float=False, cuda=self.cuda_flag).long()
                embedded_x_target = self.target_embeddings(x_target)
                dec_out, hidden_out = self.decoder_rnn(embedded_x_target, hidden[it])
                hidden[it] = hidden_out[0].detach(), hidden_out[1].detach()
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
                pred = F.tanh(t.cat([dec_out, context], -1))
                pred = self.dropout_2(pred)  # batch x target_len x 2 hdim
                pred = self.hidden2out(pred).detach()
                pred = self.lsm(pred.view(batch_size, -1)).detach()

                topk = t.topk(pred, self.beam_size, dim=1)
                top_k_indices, top_k_scores = topk[1], topk[0]
                top_k_indices = top_k_indices.transpose(0, 1)
                top_k_scores = top_k_scores.transpose(0, 1)

                for new_word_batch, new_score_batch in zip(top_k_indices, top_k_scores):
                    new_word_batch = new_word_batch.contiguous().view(batch_size, 1)
                    new_score_batch = new_score_batch.contiguous().view(batch_size, 1)

                    new_child_batch = t.cat([x_target, new_word_batch], 1).detach()

                    batch_parent_score = self.beam_scores[:, it].reshape((self.batch_size, 1))
                    batch_acc_score = batch_parent_score + new_score_batch.data.cpu().numpy()

                    if len(collective_children) > 0:
                        collective_children = np.hstack((collective_children, new_child_batch.data.cpu().numpy()))
                        # Add the corresponding beam element's score with the new score and stack it.
                        collective_scores = np.hstack((collective_scores, batch_acc_score))
                    else:
                        collective_children, collective_scores = new_child_batch.data.cpu().numpy(), batch_acc_score


                        # At the end of a for loop collective children, collective scores
            # will look a numpy array of tensors.
            current_beam_length = 1  # Means only start elem is there.
            if len(self.beam) != 1:
                current_beam_length = self.beam.shape[1]

            collective_children = collective_children.reshape((batch_size, current_beam_length * self.beam_size,
                                                               int(collective_children.shape[1] /
                                                                   current_beam_length / self.beam_size)
                                                               ))

            if collective_children.shape[1] == self.beam_size:  # Happens the first time.
                self.beam = collective_children
                self.beam_scores = collective_scores
            else:
                self.beam = deepcopy(np.zeros((batch_size, self.beam_size, collective_children.shape[2])))
                for i in range(batch_size):
                    # Since argsort gives ascending order
                    best_scores_indices = np.argsort(-1 * collective_scores[i])[:self.beam_size]
                    for key, index in enumerate(best_scores_indices):
                        self.beam[i][key][:] = collective_children[i][index]
                        self.beam_scores[i][key] = collective_scores[i][index]

            terminate_beam = True

            for x in self.beam:
                for c in x:
                    if EOS_TOKEN not in c:
                        terminate_beam = False
                        break
                if not terminate_beam:
                    break
                    # import pdb; pdb.set_trace()
            assert (self.beam.shape == (batch_size, self.beam_size, time + 2))

            time += 1
            print(time)
        return self.beam


class LSTMFP(t.nn.Module):
    """
    Implementation of `Neural Machine Translation by Jointly Learning to Align and Translate`
    https://arxiv.org/abs/1409.0473

    NOTE THAT ITS INPUT SHOULD HAVE THE BATCH SIZE FIRST !!!!!
    """

    def __init__(self, params, source_embeddings=None, target_embeddings=None):
        super(LSTMFP, self).__init__()
        print("Initializing LSTMFP")
        self.cuda_flag = params.get('cuda', CUDA_DEFAULT)
        self.model_str = 'LSTMFP'
        self.params = params

        # Initialize hyperparams.
        self.hidden_dim = params.get('hidden_dim', 100)
        self.batch_size = params.get('batch_size', 32)
        self.dropout = params.get('dropout', 0.5)
        self.embed_dropout = params.get('embed_dropout')
        self.initialize_embeddings(params, source_embeddings, target_embeddings)
        self.output_size = self.target_vocab_size
        assert self.hidden_dim == self.embedding_dim

        # Initialize network modules.
        self.encoder_rnn1 = t.nn.LSTM(self.embedding_dim, self.hidden_dim // 2, dropout=self.dropout, num_layers=1, bidirectional=True, batch_first=True)
        self.encoder_rnn2 = t.nn.LSTM(self.embedding_dim, self.hidden_dim, dropout=self.dropout, num_layers=1, bidirectional=False, batch_first=True)
        self.decoder_rnn1 = t.nn.LSTM(self.embedding_dim, self.hidden_dim, dropout=self.dropout, num_layers=1, batch_first=True)
        self.decoder_rnn2 = t.nn.LSTM(self.embedding_dim, self.hidden_dim, dropout=self.dropout, num_layers=1, batch_first=True)
        self.hidden_dec_initializer = t.nn.Linear(self.hidden_dim // 2, 2 * self.hidden_dim)
        self.hidden2out = t.nn.Linear(self.hidden_dim * 2, self.output_size)
        if self.embed_dropout:
            self.dropout_1s = t.nn.Dropout(self.dropout)
            self.dropout_1t = t.nn.Dropout(self.dropout)
        self.dropout_1_enc = t.nn.Dropout(self.dropout)
        self.dropout_1_dec = t.nn.Dropout(self.dropout)
        self.dropout_2 = t.nn.Dropout(self.dropout)
        self.linear_enc1 = t.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_enc2 = t.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_dec1 = t.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_dec2 = t.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_attn1 = t.nn.Linear(self.hidden_dim, 1, bias=False)
        self.linear_attn2 = t.nn.Linear(self.hidden_dim, 1, bias=False)
        self.hidden2dec = t.nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.lsm = nn.LogSoftmax()

        self.beam_size = params.get('beam_size', 3)
        self.max_beam_depth = params.get('max_beam_depth', 20)

        if self.cuda_flag:
            self = self.cuda()

    def initialize_embeddings(self, params, source_embeddings, target_embeddings):
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

        self.train_embedding = params.get('train_embedding', True)

        # Initialize embeddings. Static embeddings for now.
        self.source_embeddings = t.nn.Embedding(self.source_vocab_size, self.embedding_dim)
        self.target_embeddings = t.nn.Embedding(self.target_vocab_size, self.embedding_dim)
        if source_embeddings is not None:
            self.source_embeddings.weight = t.nn.Parameter(source_embeddings, requires_grad=self.train_embedding)
        if target_embeddings is not None:
            self.target_embeddings.weight = t.nn.Parameter(target_embeddings, requires_grad=self.train_embedding)

    def init_hidden(self, data, type, batch_size=None):
        """
        Initialize the hidden state, either for the encoder or the decoder

        For type=`enc`, it should just be initialized with 0s
        For type=`dec`, it should be initialized with tanh(W h1_backward) (see page 13 of the paper, last paragraph)

        `data` is either something you initialize the hidden state with, or None
        """
        bs = batch_size if batch_size is not None else self.batch_size
        if type == 'dec':
            # in that case, `data` is the output of the encoder
            # data[:, :1, self.hidden_dim // 2:]
            # `:` for the whole batch
            # `:1` because you want the hidden state of the first time step (see paper, they use backward(h1))
            # but also `self.hidden_dim // 2:`, because you want the backward part only (the last coefficients)
            h = F.tanh(self.hidden_dec_initializer(data[:, -1:, self.hidden_dim // 2:]))  # the last hdim/2 weights correspond to the backward layer(s)
            h = h.transpose(1, 0)
            h = t.cat(t.split(h, self.hidden_dim, dim=2), 0)
            return (
                h,
                variable(np.zeros((2, bs, self.hidden_dim)), cuda=self.cuda_flag)
            )
        elif type == 'enc1':
            # in that case data is None
            return tuple((
                variable(np.zeros((2, bs, self.hidden_dim // 2)), cuda=self.cuda_flag),
                variable(np.zeros((2, bs, self.hidden_dim // 2)), cuda=self.cuda_flag)
            ))
        elif type == 'enc2':
            # in that case data is None
            return tuple((
                variable(np.zeros((1, bs, self.hidden_dim)), cuda=self.cuda_flag),
                variable(np.zeros((1, bs, self.hidden_dim)), cuda=self.cuda_flag)
            ))
        else:
            raise ValueError('the type should be either `dec` or `enc`')

    def forward(self, x_source, x_target, return_attn=False):
        batch_size = x_source.size(0)
        src_len = x_source.size(1)
        trg_len = x_target.size(1) - 1

        # EMBEDDING
        embedded_x_source = self.source_embeddings(x_source)
        embedded_x_target = self.target_embeddings(x_target[:, :-1])  # don't make a prediction for the word following the last one
        if self.embed_dropout:
            embedded_x_source = self.dropout_1s(embedded_x_source)
            embedded_x_target = self.dropout_1t(embedded_x_target)

        # RECURRENT: 2 layers of encoder and 2 layers of decoder. There is dropout inbetween layers, and skip connections as well
        # ENCODER
        hidden1 = self.init_hidden(None, 'enc1', batch_size)
        hidden2 = self.init_hidden(None, 'enc2', batch_size)
        enc_out, _ = self.encoder_rnn1(embedded_x_source, hidden1)
        enc_out, _ = self.encoder_rnn2(self.dropout_1_enc(embedded_x_source + enc_out), hidden2)  # skip connection + dropout

        # DECODER
        # Hidden states init
        hidden12 = self.init_hidden(enc_out, 'dec', batch_size)
        hidden1 = hidden12[0][:1, :, :], hidden12[1][:1, :, :]
        hidden2 = hidden12[0][1:, :, :], hidden12[1][1:, :, :]
        # Layer 1
        dec_out, _ = self.decoder_rnn1(embedded_x_target, hidden1)
        # Attention
        context, _ = self.attend(enc_out, dec_out, batch_size, src_len, trg_len, 1, False)
        dec_out = t.cat([dec_out, context], -1)  # batch x target_len x 2 hdim
        dec_out = self.dropout_1_dec(F.tanh(self.hidden2dec(dec_out)))  # batch x target_len x hdim
        # Layer 2
        dec_out, _ = self.decoder_rnn2(embedded_x_target + dec_out, hidden2)
        # Attention
        context, attn_dist = self.attend(enc_out, dec_out, batch_size, src_len, trg_len, 2, return_attn)

        # OUTPUT
        pred = t.cat([dec_out, context], -1)
        pred = self.dropout_2(pred)  # batch x target_len x 2 hdim
        pred = self.hidden2out(pred)

        if return_attn:
            return pred, attn_dist
        else:
            return pred

    def attend(self, enc_out, dec_out, batch_size, src_len, trg_len, id, return_attn):
        # ATTENTION: just like in the paper
        linear_attn = getattr(self, 'linear_attn'+str(id))
        linear_enc = getattr(self, 'linear_enc'+str(id))
        linear_dec = getattr(self, 'linear_dec'+str(id))
        scores = linear_attn(F.tanh(
            linear_enc(enc_out).unsqueeze(2).expand(batch_size, src_len, trg_len, self.hidden_dim) +
            linear_dec(dec_out).unsqueeze(1).expand(batch_size, src_len, trg_len, self.hidden_dim)
        )).squeeze(3)
        attn_dist = F.softmax(scores, dim=1)  # batch x source_len x target_len
        context = t.bmm(attn_dist.permute(0, 2, 1), enc_out)  # batch x target_len x hidden_dim
        if return_attn:
            return context, attn_dist
        else:
            return context, None

    def translate(self, x_source):
        batch_size = x_source.size(0)
        src_len = x_source.size(1)
        self.eval()

        # EMBEDDING
        embedded_x_source = self.source_embeddings(x_source)
        if self.embed_dropout:
            embedded_x_source = self.dropout_1s(embedded_x_source)

        # RECURRENT
        hidden1 = self.init_hidden(None, 'enc1', batch_size)
        hidden2 = self.init_hidden(None, 'enc2', batch_size)
        enc_out, _ = self.encoder_rnn1(embedded_x_source, hidden1)
        enc_out, _ = self.encoder_rnn2(self.dropout_1_enc(embedded_x_source + enc_out), hidden2)  # skip connection + dropout
        x_target = (SOS_TOKEN * t.ones(x_source.size(0), 1)).long()  # `2` is the SOS token (<s>)
        x_target = variable(x_target, to_float=False, cuda=self.cuda_flag)
        count_eos = 0
        time = 0
        while count_eos < x_source.size(0):
            embedded_x_target = self.target_embeddings(x_target)
            hidden12 = self.init_hidden(enc_out, 'dec', batch_size)
            hidden1 = hidden12[0][:1, :, :], hidden12[1][:1, :, :]
            hidden2 = hidden12[0][1:, :, :], hidden12[1][1:, :, :]
            dec_out, _ = self.decoder_rnn1(embedded_x_target, hidden1)
            dec_out, _ = self.decoder_rnn2(self.dropout_1_dec(embedded_x_target + dec_out), hidden2)
            dec_out = dec_out[:, time:time + 1, :].detach()

            # ATTENTION: just like in the paper
            scores = self.linear_attn(F.tanh(
                self.linear_enc(enc_out).unsqueeze(2).expand(batch_size, src_len, 1, self.hidden_dim) +
                self.linear_dec(dec_out).unsqueeze(1).expand(batch_size, src_len, 1, self.hidden_dim)
            )).squeeze(3)
            attn_dist = F.softmax(scores, dim=1)  # batch x source_len x target_len
            context = t.bmm(attn_dist.permute(0, 2, 1), enc_out)  # batch x target_len x hidden_dim

            # OUTPUT
            # concatenate the output of the decoder and the context and apply nonlinearity
            pred = F.tanh(t.cat([dec_out, context], -1))
            pred = self.dropout_2(pred)  # batch x target_len x 2 hdim
            pred = self.hidden2out(pred).detach()
            x_target = t.cat([x_target, pred.max(2)[1]], 1).detach()

            # should you stop ?
            count_eos += t.sum((pred.max(2)[1] == EOS_TOKEN).long()).data.cpu().numpy()[0]  # `3` is the EOS token
            time += 1
        return x_target

    # @todo: implement this
    def translate_beam(self, x_source, print_beam_row=-1):
        self.eval()

        # EMBEDDING
        embedded_x_source = self.source_embeddings(x_source)
        if self.embed_dropout:
            embedded_x_source = self.dropout_1s(embedded_x_source)

        terminate_beam = False
        batch_size = x_source.size(0)

        # RECURRENT
        hidden = self.init_hidden(None, 'enc', x_source.size(0))
        enc_out, _ = self.encoder_rnn(embedded_x_source, hidden)

        # One hidden for each beam element.
        hidden = []
        for i in range(self.beam_size):
            hidden.append(self.init_hidden(enc_out, 'dec', x_source.size(0)))

        x_target = SOS_TOKEN * np.ones((x_source.size(0), 1))  # `2` is the SOS token (<s>)
        count_eos = 0
        time = 0

        # INIT SOME STUFF.
        self.beam = np.array([x_target])
        self.beam_scores = np.zeros((batch_size, 1))

        while not terminate_beam and time < self.max_beam_depth:
            collective_children = np.array([])
            collective_scores = np.array([])

            if len(self.beam) == 1:
                reshaped_beam = self.beam
            else:
                reshaped_beam = np.transpose(self.beam, (1, 0, 2))

            for it, elem in enumerate(reshaped_beam):
                elem = t.from_numpy(elem).long()
                x_target = elem.contiguous().view(self.batch_size, -1)
                x_target = variable(x_target, to_float=False, cuda=self.cuda_flag).long()
                embedded_x_target = self.target_embeddings(x_target)
                dec_out, hidden_out = self.decoder_rnn(embedded_x_target, hidden[it])
                hidden[it] = hidden_out[0].detach(), hidden_out[1].detach()
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
                pred = F.tanh(t.cat([dec_out, context], -1))
                pred = self.dropout_2(pred)  # batch x target_len x 2 hdim
                pred = self.hidden2out(pred).detach()
                pred = self.lsm(pred.view(batch_size, -1)).detach()

                topk = t.topk(pred, self.beam_size, dim=1)
                top_k_indices, top_k_scores = topk[1], topk[0]
                top_k_indices = top_k_indices.transpose(0, 1)
                top_k_scores = top_k_scores.transpose(0, 1)

                for new_word_batch, new_score_batch in zip(top_k_indices, top_k_scores):
                    new_word_batch = new_word_batch.contiguous().view(batch_size, 1)
                    new_score_batch = new_score_batch.contiguous().view(batch_size, 1)

                    new_child_batch = t.cat([x_target, new_word_batch], 1).detach()

                    batch_parent_score = self.beam_scores[:, it].reshape((self.batch_size, 1))
                    batch_acc_score = batch_parent_score + new_score_batch.data.cpu().numpy()

                    if len(collective_children) > 0:
                        collective_children = np.hstack((collective_children, new_child_batch.data.cpu().numpy()))
                        # Add the corresponding beam element's score with the new score and stack it.
                        collective_scores = np.hstack((collective_scores, batch_acc_score))
                    else:
                        collective_children, collective_scores = new_child_batch.data.cpu().numpy(), batch_acc_score


                        # At the end of a for loop collective children, collective scores
            # will look a numpy array of tensors.
            current_beam_length = 1  # Means only start elem is there.
            if len(self.beam) != 1:
                current_beam_length = self.beam.shape[1]

            collective_children = collective_children.reshape((batch_size, current_beam_length * self.beam_size,
                                                               int(collective_children.shape[1] /
                                                                   current_beam_length / self.beam_size)
                                                               ))

            if collective_children.shape[1] == self.beam_size:  # Happens the first time.
                self.beam = collective_children
                self.beam_scores = collective_scores
            else:
                self.beam = deepcopy(np.zeros((batch_size, self.beam_size, collective_children.shape[2])))
                for i in range(batch_size):
                    # Since argsort gives ascending order
                    best_scores_indices = np.argsort(-1 * collective_scores[i])[:self.beam_size]
                    for key, index in enumerate(best_scores_indices):
                        self.beam[i][key][:] = collective_children[i][index]
                        self.beam_scores[i][key] = collective_scores[i][index]

            terminate_beam = True

            for x in self.beam:
                for c in x:
                    if EOS_TOKEN not in c:
                        terminate_beam = False
                        break
                if not terminate_beam:
                    break
                    # import pdb; pdb.set_trace()
            assert (self.beam.shape == (batch_size, self.beam_size, time + 2))

            time += 1
            print(time)
        return self.beam


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

        >>> loss = TemporalCrossEntropyLoss()
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
        Let `T` be the sentence length and `|V|` the vocab size (#classes)
        What this class does is just reshaping the inputs so that you can use classical cross entropy on top of that

        :param pred: FloatTensor of shape (batch_size, |V|, T)
        :param true: LongTensor of shape (batch_size, T)
        :return:
        """
        t.nn.modules.loss._assert_no_grad(true)

        # doing it this way allows to use parallelism. Better than looping on last dim !
        # note that this version of pytorch seems outdated
        true_ = true.contiguous().view(true.size(0) * true.size(1))  # true_.size() = (batch_size*T, )
        pred_ = pred.contiguous().view(pred.size(0) * pred.size(2), pred.size(1))  # pred_.size() = (batch_size*T, |V|)
        return self.cross_entropy.forward(pred_, true_)