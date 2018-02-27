# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 22:31:49 2018

@author: SrivatsanPC
"""
import torchtext
from torchtext.vocab import Vectors, GloVe, FastText
from utils import variable
from const import *
import numpy as np
from torch.autograd import Variable
import spacy
from torchtext import data
from torchtext import datasets
import pickle


def generate_iterators(BATCH_SIZE=32, MAX_LEN=20, load_data=False, embedding=None):
    if not load_data:
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

        train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(DE, EN),
                                                 filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
                                                                       len(vars(x)['trg']) <= MAX_LEN)
        MIN_FREQ = 5
        DE.build_vocab(train.src, min_freq=MIN_FREQ)
        EN.build_vocab(train.trg, min_freq=MIN_FREQ)
        if embedding is not None:
            if embedding in ['FastText', 'fasttext']:
                EN.vocab.load_vectors(vectors=FastText(language='en'))
                DE.vocab.load_vectors(vectors=FastText(language='de'))
            else:
                raise ValueError("Only fasttext is supported at the moment")
        train_iter, val_iter = data.BucketIterator.splits((train, val), batch_size=BATCH_SIZE, device=-1,
                                                          repeat=False, sort_key=lambda x: len(x.src))

        return train_iter, val_iter, EN, DE
    else:  # does not work...
        with open('train.pkl', 'rb') as f:
            train = pickle.load(f)
        with open('val.pkl', 'rb') as f:
            val = pickle.load(f)
        with open('DE.torchtext.Field.pkl', 'rb') as f:
            DE = pickle.load(f)
        with open('EN.torchtext.Field.pkl', 'rb') as f:
            EN = pickle.load(f)
        BATCH_SIZE = 32
        train_iter, val_iter = data.BucketIterator.splits((train, val), batch_size=BATCH_SIZE, device=-1,
                                                          repeat=False, sort_key=lambda x: len(x.src))
        return train_iter, val_iter, EN, DE


def escape(l):
    return l.replace("\"", "<quote>").replace(",", "<comma>")


# @todo: implement this
def generate_kaggle_text(val_iter, trained_model, EN, batch_size, beam_size, total_sentences, num_words=3, expt_name="LSTM_Attention", debug=False, print_on_screen=False):
    batch_count = 0
    top_predictions = np.ones((total_sentences, beam_size, num_words)) * -1

    for batch in val_iter:
        pred_beam = trained_model.translate_beam(batch.src.transpose(0, 1).cuda())

        for i in range(batch_size):
            for j in range(beam_size):
                # import pdb; pdb.set_trace()
                top_predictions[batch_count * batch_size + i, j] = pred_beam[i, j, 1:num_words + 1]
        batch_count += 1

        if debug:
            break

    if not print_on_screen:
        with open(expt_name + ".txt", "w") as fout:
            print("id,word")
            for i in range(total_sentences):
                print(str(i + 1) + ",", end="")
                for j in range(beam_size):
                    print("|".join([escape(EN.vocab.itos[int(x)]) for x in top_predictions[i][j]]), " ", end="")
                print("")
    else:
        print("id,word")
        for i in range(total_sentences):
            print(str(i + 1) + ",", end="")
            for j in range(beam_size):
                print("|".join([escape(EN.vocab.itos[int(x)]) for x in top_predictions[i][j]]), " ", end="")
            print("")
