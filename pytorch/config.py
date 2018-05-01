# -*- coding: utf-8 -*-
"""
Created on Tue May  1 12:06:40 2018

@author: SrivatsanPC
"""

POS_Map = {'Noun': ['NN','NNS','NNP', 'NNPS' ],
           'Verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
           'Adverb' : ['RB', 'RBR', 'RBS', 'WRB'],
           'Adjective' : ['JJ', 'JJR', 'JJS', ],
           'Pronoun' : ['PRP', 'PRP$', 'WP', 'WP$'],
           'Others': ['LS', 'FW', 'CD', 'PDT', 'RP', 'UH'],
           'Misc' :  ['WDT', 'TO', 'POS', 'MD', 'IN', 'DT', 'EX', 'CC']
}

def get_reverse_POS_Map(POS_Map):
    reverse_POS_Map = {}
    for key in POS_Map.keys():
        vals = POS_Map[key]
        for v in list(vals):
            reverse_POS_Map[v] = key
    return reverse_POS_Map

Key_to_POS_Map = get_reverse_POS_Map(POS_Map)

POS_Schedule = [
{'Noun' :1000, 'Verb' : 500, 'Adjective' :200, 'Adverb' :50, 'Misc':30, 'Pronoun': 10, 'Others':0},
{'Noun' :3000, 'Verb' : 1000, 'Adjective' :400, 'Adverb' :100, 'Misc':50, 'Pronoun': 20, 'Others':30},
{'Noun' :5000, 'Verb' : 1700, 'Adjective' :800, 'Adverb' :200, 'Misc':100, 'Pronoun': 44, 'Others':97},
['all']
]
    
prog_vocab_list = [1000, 2000, 5000, 8000, 11000]
