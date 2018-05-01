# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 12:02:37 2018

@author: SrivatsanPC
"""

from matplotlib import pyplot as plt
import pickle

exp13 = pickle.load(open('exp_13_BLEU.p', 'rb'))
exp13b = pickle.load(open('exp_13b_BLEU.p', 'rb'))
exp22 = pickle.load(open('exp_22_BLEU.p', 'rb'))
exp23 = pickle.load(open('exp23_BLEU.p', 'rb'))

count = 0
names = ['WGAN-GP-NP', 'WGAN-WC-UN', 'WGAN-GP-UN']
for expt in [exp23, exp13b, exp22]:
    BLEU_1 = [i[0] for i in list(expt.values())]
    BLEU_2 = [i[1] for i in list(expt.values())]
    BLEU_3 = [i[2] for i in list(expt.values())]
    BLEU_4 = [i[3] for i in list(expt.values())]  
  

    plt.plot(list(expt.keys()), list(BLEU_1), label = 'BLEU1')
    plt.plot(list(expt.keys()), list(BLEU_2), label = 'BLEU2')
    plt.plot(list(expt.keys()), list(BLEU_3), label = 'BLEU3')
    plt.plot(list(expt.keys()), list(BLEU_4), label = 'BLEU4(Standard BLEU)')
    plt.title('BLEU scores of '+ names[count] +' . Note BLEU-N is the BLEU over all n-grams upto N')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    count += 1
    plt.show()
    
for i in range(4):
    count = 0
    for expt in [exp23, exp13b, exp22]:
        BLEU = [p[i] for p in list(expt.values())]
        plt.plot(list(expt.keys()), list(BLEU), label = names[count])
        count += 1
    plt.title('BLEU - ' +str(i+1) + ' scores')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    