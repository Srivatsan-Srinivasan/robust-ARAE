# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 00:20:59 2018

@author: SrivatsanPC
"""

from nltk.tokenize import TweetTokenizer
from random import sample
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
import pickle
from copy import deepcopy

tknzr = TweetTokenizer()

def return_length_dict(content, gen = False) :
    corpus_as_dict_sent = {}
    corpus_as_dict_tokens ={}
    for sentence in tqdm(content):
       tokens = tknzr.tokenize(sentence)
       length = len(tokens)
       if length <= 30:
           if length in corpus_as_dict_sent:
               corpus_as_dict_sent[length].append(sentence)
               corpus_as_dict_tokens[length].append(tokens)
           else:
               corpus_as_dict_sent[length] = [sentence]
               corpus_as_dict_tokens[length] = [tokens]
           
    return corpus_as_dict_sent, corpus_as_dict_tokens

#NLTK Tutorial : https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
def calculate_corpus_bleu_by_length(test_tokens_as_dict, generation_tokens_as_dict, save_filename = 'dummy'):
    corpus_bleu_by_length = {}
    min_len = 4
    max_len = max(generation_tokens_as_dict.keys())
    references_overall = []
    
    print("Maximum length of generated tokens is ", max_len )
    
    for i in list(test_tokens_as_dict.values()):
        if len(references_overall):
            references_overall = references_overall + i
        else:
            references_overall = deepcopy(i)
        
    for leng in tqdm(range(min_len,max_len)):
        if leng in generation_tokens_as_dict:
            hypotheses = generation_tokens_as_dict[leng]
            references = [deepcopy(references_overall) for i in range(len(hypotheses))]
            BLEU_1 = corpus_bleu(references, hypotheses, weights = (1,0,0,0))
            BLEU_2 = corpus_bleu(references, hypotheses, weights = (0.5,0.5,0,0))
            BLEU_3 = corpus_bleu(references, hypotheses, weights = (0.333,0.333,0.334,0))
            BLEU_4 = corpus_bleu(references, hypotheses, weights = (0.25,0.25,0.25,0.25))
            corpus_bleu_by_length[leng] = [BLEU_1,BLEU_2,BLEU_3,BLEU_4]

    pickle.dump(corpus_bleu_by_length, open( save_filename + "_BLEU.p", "wb" )  )

test_snli_file = '../snli_lm/test.txt'
train_snli_file = '../snli_lm/train.txt'
exp13 = '../generated_data/generated_exp13.txt'
exp13b = '../generated_data/generated_exp13b.txt'
exp22 = '../generated_data/generated_exp22.txt'
exp23 = '../generated_data/generated_exp23ddd_63ppl.txt'
exp25 = '../generated_data/generated_exp25.txt'

generated_Files = {'exp13':exp13, 'exp13b':exp13b, 'exp22':exp22, 'exp23': exp23, 'exp25':exp25}

with open(test_snli_file) as f:
    sample_content_test = f.readlines()
with open(train_snli_file) as f:
    sample_content_train = f.readlines()
#sample_content = sample_content_test + sample_content_train
sample_content = sample_content_test
# you may also want to remove whitespace characters like `\n` at the end of each line
test_samples = [x.strip() for x in sample_content ] 
#import pdb; pdb.set_trace()
test_sent_as_dict, test_tokens_as_dict = return_length_dict(test_samples)

test_number_map = {}
for i in list(test_tokens_as_dict.keys()):
    value = test_tokens_as_dict[i]
    num = min(100000, len(value))
    test_tokens_as_dict[i] = sample(value, num)
    test_number_map[i] = num
pickle.dump(test_number_map, open('test_data_numbers.p', "wb"))

name_map = {exp13: 'exp_13', exp22  :'exp_22', exp13b : 'exp_13b', exp23 : 'exp_23', exp25 : 'exp_25'}

for experiment in [exp22, exp23, exp25, exp13b]:
    with open(experiment) as f:
        gen_content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    generated_samples = [x.strip() for x in gen_content] 
    generation_sent_as_dict, generation_tokens_as_dict = return_length_dict(generated_samples, gen = True)
    generation_number_map = {}
    
    for i in list(generation_tokens_as_dict.keys()):
        value = generation_tokens_as_dict[i]        
        num = min(1000, len(value))
        generation_tokens_as_dict[i] = sample(value, num)
        generation_number_map[i] = num
    pickle.dump(generation_number_map, open('gen_data_numbers_' + name_map[experiment] +'.p', "wb"))
       
    calculate_corpus_bleu_by_length(test_tokens_as_dict, generation_tokens_as_dict, save_filename = name_map[experiment])
    print(name_map[experiment] + ' over.')


bleu_scores = {}


