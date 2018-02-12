# Text text processing library
import torchtext
from torchtext.vocab import Vectors

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
torch.manual_seed(1)

import numpy as np
from scipy.optimize import minimize
import math

#PREPARING DATA
TEXT = torchtext.data.Field()
# Data distributed with the assignment
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
    path=".", 
    train="train.txt", validation="valid.txt", test="valid.txt", text_field=TEXT)

TEXT.build_vocab(train)
train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
    (train, val, test), batch_size=10, device=-1, bptt_len=32, repeat=False)


if False: #do this to find optimal alpha
    q_trigram_valid = np.zeros([vocab_size,vocab_size,vocab_size],dtype=int)
    for batch in iter(val_iter):
        for j in range(10):
            sample = batch.text[:,j]
            s = len(sample)
            for i in range(s):
                if i < s - 3: #then we can still do trigram
                    q_trigram_valid[sample[i].data,sample[i+1].data,sample[i+2].data] += 1

    def L(alpha):
        loss = 0
        for batch in iter(val_iter):
            for j in range(10):
                sample = batch.text[:,j]
                s = len(sample)
                for i in range(s):
                    if i < s - 3: #then we can still do trigram
                        c = q_trigram_valid[sample[i].data,sample[i+1].data,sample[i+2].data]
                        q = 1.0*alpha[0]*float(q_trigram[sample[i].data,sample[i+1].data,sample[i+2].data])/float(q_bigram[sample[i].data,sample[i+1].data])
                        q += 1.0*alpha[1]*float(q_bigram[sample[i+1].data,sample[i+2].data])/float(q_unigram[sample[i+1].data])
                        q += 1.0*(1-alpha[0]-alpha[1])*float(q_unigram[sample[i+2].data])/total_number_of_words
                        if q and not math.isnan(q) and not q == float('Inf'):
                            l = math.log(q)
                            loss += float(c)*l
                        #print loss
        print -loss,alpha
        return -loss

    cons = ({'type': 'ineq',
             'fun' : lambda x: np.array([x[0]])},
            {'type': 'ineq',
             'fun' : lambda x: np.array([1-x[0]])},
            {'type': 'ineq',
             'fun' : lambda x: np.array([x[1]])},
            {'type': 'ineq',
             'fun' : lambda x: np.array([1-x[1]])})

    res = minimize(L, [1.0/3,1.0/3], constraints=cons, method='nelder-mead', options={'disp': True})


#LEARNING MLE

vocab_size = 10001
total_number_of_words = 0.0
batch_size = 10
bptt_len=32

q_unigram = np.zeros(vocab_size,dtype=int)
q_bigram = np.zeros([vocab_size,vocab_size],dtype=int)
q_trigram = np.zeros([vocab_size,vocab_size,vocab_size],dtype=int)
counter = 0

#concat all batched data to one list
full_train_dataset = []
for batch in iter(train_iter):
    counter += 1
    if counter % 1000 == 0:
        print("#",counter)
    for j in range(batch_size):
        full_train_dataset = full_train_dataset + list(batch.text[:,j].data)

total_number_of_words = len(full_train_dataset)
        
#learn uni-bi-trigrams
for pos in range(total_number_of_words):
    q_unigram[full_train_dataset[pos]] += 1
    if pos < total_number_of_words - 2: #then we can still do bigram
        q_bigram[full_train_dataset[pos],full_train_dataset[pos+1]] += 1
    if pos < total_number_of_words - 3: #then we can still do trigram
        q_trigram[full_train_dataset[pos],full_train_dataset[pos+1],full_train_dataset[pos+2]] += 1

#SAMPLING DATA

alpha = [0.559375, 0.39479167]

if False: #if you want to give equal weight to uni-bi-trigrams
alpha = [1.0/3,1.0/3]

def trigram_interpolation_model(sample):
    probabilities = []
    for j in range(len(TEXT.vocab)):
        try:
            q = 1.0*alpha[0]*float(q_trigram[sample[-2],sample[-1],j])/float(q_bigram[sample[-2],sample[-1]])
        except:
            q=0.0
        try:
            q += 1.0*alpha[1]*float(q_bigram[sample[-1],j])/float(q_unigram[sample[-1]])
        except:
            temp = 0
        q += 1.0*(1-alpha[0]-alpha[1])*float(q_unigram[j])/total_number_of_words
        if j == TEXT.vocab.stoi["<eos>"]:
            q = 0.0
        probabilities.append(q)
    return probabilities

def generate_kaggle_output(model,input_file="input.txt", output_file="output.txt"):
    #test_iter is the test iterator from penn
    #model is a function that returns probability distribution for next given input string (list of indicies)
    id_sample = 0
    with open(output_file, "w") as fout: 
        with open(input_file) as fin:
            for sentence in fin.readlines():
                sample = [TEXT.vocab.stoi[s] for s in sentence.strip().split() if not "__" in s]
                probs = model(sample)
                id_sample += 1
                ix_predictions = sorted(range(len(probs)), key=lambda i: probs[i],reverse=True)[:20]
                predictions = [TEXT.vocab.itos[i] for i in ix_predictions]
                out = str(id_sample) + "," + " ".join(predictions) + "\n"
                fout.write(out)

generate_kaggle_output(trigram_interpolation_model)