# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 22:31:49 2018

@author: SrivatsanPC
"""
import torchtext
from torchtext.vocab import Vectors

def generate_iterators(debug = False, batch_size = 10):
    TEXT = torchtext.data.Field()
    # Data distributed with the assignment
    train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
    path=".", 
    train="train.txt", validation="valid.txt", test="valid.txt", text_field=TEXT)
    if debug:
        TEXT.build_vocab(train, max_size=1000)
        print('len(TEXT.vocab)', len(TEXT.vocab))
        
    TEXT.build_vocab(train)
    print('len(TEXT.vocab)', len(TEXT.vocab))
    
    train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
    (train, val, test), batch_size=batch_size, device=-1, bptt_len=32, repeat=False, shuffle= False)
    
    return train_iter, val_iter, test_iter, TEXT, len(TEXT.vocab)
    