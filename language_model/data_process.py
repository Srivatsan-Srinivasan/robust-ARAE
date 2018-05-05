# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 22:31:49 2018

@author: SrivatsanPC
"""
import torchtext
from torchtext.vocab import Vectors, GloVe
from utils import variable, Corpus, batchify
from const import *
import numpy as np
from torch.autograd import Variable


def generate_iterators(args):
    # Data distributed with the assignment

    corpus = Corpus(args.data_path,
                    maxlen=args.maxlen,
                    vocab_size=args.vocab_size,
                    lowercase=True)

    test_data = batchify(corpus.test, 10, shuffle=False, gpu_id=args.gpu_id)
    train_data = batchify(corpus.train, args.batch_size, shuffle=True, gpu_id=args.gpu_id)

    return train_data, test_data, corpus


def generate_text(trained_model, expt_name, TEXT, context_size=None, n=20, cuda=CUDA_DEFAULT, h_dim=100):
    #Sentences processed one at a time.
    trained_model.eval()
    hidden = tuple((
                variable(np.zeros((trained_model.num_layers, 1, trained_model.hidden_dim)), cuda=cuda),
                variable(np.zeros((trained_model.num_layers, 1, trained_model.hidden_dim)), cuda=cuda)
            ))
    with open(expt_name + ".txt", "w") as fout:
        print("id,word\n", file=fout)
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
                       
            hidden_init = hidden[0].data
            memory_init = hidden[1].data
            
            if trained_model.model_str in recur_models:
                trained_model.zero_grad()
                trained_model.hidden = (Variable(hidden_init), Variable(memory_init))
            output, hidden = trained_model(x_test.long())            

            # Batch * NO of words * vocab

            output = output.view(1, len(word_markers), -1)
            if cuda:
                output = output.data.cpu().numpy()
            else:
                output = output.data.numpy()
                
            output = output[0]

            # top 20 predicitons for Last word
            n_predictions = (-output[-1]).argsort()[:21]            
            predictions = [TEXT.vocab.itos[i] for i in n_predictions]
            if '<eos>' in predictions:
                predictions.remove('<eos>')
            else:
                predictions = predictions[:20]
                
            print("%d,%s" % (i, " ".join(predictions)), file=fout)
        print("Completed writing the output file successfully")
