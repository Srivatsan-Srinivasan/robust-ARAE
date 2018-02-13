# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 22:31:49 2018

@author: SrivatsanPC
"""
import torchtext
from torchtext.vocab import Vectors, GloVe
from utils import variable
from const import *
import numpy as np
from torch.autograd import Variable


def generate_iterators(model_str, debug=False, batch_size=10, emb='GloVe', context_size=None, emb_size = 50):
    TEXT = torchtext.data.Field()
    # Data distributed with the assignment

    train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
        path="../HW2/",
        train="train.txt", validation="valid.txt", test="valid.txt", text_field=TEXT)
    if debug:
        TEXT.build_vocab(train, max_size=1000)
        print('len(TEXT.vocab)', len(TEXT.vocab))

    TEXT.build_vocab(train)
    print('len(TEXT.vocab)', len(TEXT.vocab))

    if model_str != 'NNLM2':
        train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
            (train, val, test), batch_size=batch_size, device=-1, bptt_len=32, repeat=False, shuffle=False)
        train_iter.shuffle = False
        val_iter.shuffle = False
    else:
        """
        In that case the dataset consists of contexts + next_word only. it has to be chopped of
        The train_iter, val_iter and test_iter are lists of numpy array in this case, of shape (1, context_size+1) (+1 because it includes the word to be predicted)
        """
        train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
            (train, val, test), batch_size=1, device=-1, bptt_len=1000000, repeat=False)

        # Verifications
        assert context_size is not None, 'context_size should be an integer'
        assert len(train_iter) == 1, " there should be only one batch... increase bptt_len"
        assert len(val_iter) == 1, " there should be only one batch... increase bptt_len"
        assert len(test_iter) == 1, " there should be only one batch... increase bptt_len"

        # Chopping off datasets
        for batch in train_iter:
            dataset = []
            for k in range(batch.text.size(0) - context_size - 1):
                dataset.append(batch.text.squeeze()[k:k + context_size + 1].data.numpy().reshape((1, context_size + 1)))
        train_iter = dataset
        for batch in val_iter:
            dataset = []
            for k in range(batch.text.size(0) - context_size - 1):
                dataset.append(batch.text.squeeze()[k:k + context_size + 1].data.numpy().reshape((1, context_size + 1)))
        val_iter = dataset
        for batch in test_iter:
            dataset = []
            for k in range(batch.text.size(0) - context_size - 1):
                dataset.append(batch.text.squeeze()[k:k + context_size + 1].data.numpy().reshape((1, context_size + 1)))
        test_iter = dataset

    # Load pre-trained word embddings if any
    if emb == 'GloVe':
        TEXT.vocab.load_vectors(vectors=GloVe(name='6B', dim=emb_size))
        print("Embedding GloVe on to dimensions = " + str(emb_size))
    elif emb == 'fasttext':
        url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
        TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))

    return train_iter, val_iter, test_iter, TEXT, len(TEXT.vocab), TEXT.vocab.vectors


def generate_text(trained_model, expt_name, TEXT, context_size=None, n=20, cuda=CUDA_DEFAULT, h_dim=100):
    with open(expt_name + ".txt", "w") as fout:
        print("id,word", file=fout)
        for i, l in enumerate(open("input.txt"), 1):
            if trained_model.model_str == 'NNLM2':
                assert context_size is not None, '`context_size` should be an integer'
                assert isinstance(context_size, int), '`context_size` should be an integer'
                word_markers = [TEXT.vocab.stoi[s] for s in l.split()][-context_size - 1:-1]
            else:
                word_markers = [TEXT.vocab.stoi[s] for s in l.split()][:-1]

            # Input format to the model. Batch_size * bptt.
            # for now, batch_size = 1.
            x_test = variable(np.matrix(word_markers), requires_grad=False, cuda=cuda)
            hidden = trained_model.init_hidden()
            if trained_model.model_str in recur_models:
                trained_model.zero_grad()
                trained_model.hidden = (Variable(hidden[0].detach), Variable(hidden[1].detach()))
            output, hidden = trained_model(x_test.long())            

            # Batch * NO of words * vocab

            output = output.view(1, len(word_markers), -1)
            if cuda:
                output = output.data.cpu().numpy()
            else:
                output = output.data.numpy()
                
            output = output[0]

            # top 20 predicitons for Last word
            n_predictions = (-output[-1]).argsort()[:20]
            print("%d,%s" % (i, " ".join(n_predictions)), file=fout)
        print("Completed writing the output file successfully")
