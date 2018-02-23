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
import spacy
from torchtext import data
from torchtext import datasets


def generate_iterators():
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    DE = data.Field(tokenize=tokenize_de)
    EN = data.Field(tokenize=tokenize_en, init_token=BOS_WORD, eos_token=EOS_WORD)  # only target needs BOS/EOS

    MAX_LEN = 20
    train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(DE, EN),
                                             filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
                                                                   len(vars(x)['trg']) <= MAX_LEN)
    MIN_FREQ = 5
    DE.build_vocab(train.src, min_freq=MIN_FREQ)
    EN.build_vocab(train.trg, min_freq=MIN_FREQ)
    BATCH_SIZE = 32
    train_iter, val_iter = data.BucketIterator.splits((train, val), batch_size=BATCH_SIZE, device=-1,
                                                      repeat=False, sort_key=lambda x: len(x.src))

    return train_iter, val_iter, EN, DE


# @todo: implement this
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
