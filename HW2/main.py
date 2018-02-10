# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 15:25:48 2018

@author: SrivatsanPC
"""

#Mostly dummy code for now to illustrate argparser.
import argparse
from validate_params import check_args
from const import *
#Create Parser.
parser = argparse.ArgumentParser(description = "For CS287 HW2")

#Add arguments to be parsed.
parser.add_argument('--model', default = 'Trigram', help = 'state which model to use')
parser.add_argument('--debug', default = False )
parser.add_argument('--cuda', default = True)
parser.add_argument('--optimizer', default = 'SGD')
parser.add_argument('--lstm_n_L', default = 1)
parser.add_argument('--lstm_h_dim', default = 100)
parser.add_argument('--embed_size', default = 300)
parser.add_argument('--batch_size', default = 10)
parser.add_argument('--dropout', default = 0.5)
parser.add_argument('--lr', default = 0.1)
parser.add_argument('--exp_n', default = 'dummy_expt', help = 'Give name for expt')
parser.add_argument('--save', default = False, help = 'States if you need to pickle validation loss')

#Actually Parse. After this , any argument could be accessed by args.<argument_name>.Also validate.
args = parser.parse_args()
check_args()
    
#Load data code should be here.


#Call for different models code should be here.


#Dummy code.
print("The model is ", args.model)