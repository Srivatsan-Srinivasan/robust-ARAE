# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 00:20:59 2018

@author: SrivatsanPC
"""

from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()
with open("sample_corpus.txt") as f:
    corpus_content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
corpus_content = [x.strip() for x in corpus_content] 

with open("sample_generation.txt") as f:
    generation_content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
generation_content = [x.strip() for x in generation_content ] 


def return_length_dict(content) :
    corpus_as_dict = {}
    for sentence in content:
       length = len(tknzr.tokenize(sentence))
       if length in corpus_as_dict:
           corpus_as_dict[length] = corpus_as_dict[length] + [sentence]
       else:
           corpus_as_dict[length] = [sentence]
    return corpus_as_dict

corpus_as_dict = return_length_dict(corpus_content)
generation_as_dict = return_length_dict(generation_content)

bleu_scores = {}


