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
    
