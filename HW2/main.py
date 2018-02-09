# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 15:25:48 2018

@author: SrivatsanPC
"""

#Mostly dummy code for now to illustrate argparser.
import argparse
from const import *
#Create Parser.
parser = argparse.ArgumentParser(description = "For CS287 HW2")

#Add arguments to be parsed.
parser.add_argument('--model', default = 'Trigram', help = 'state which model to use')
parser.add_argument('--debug', default = False )

#Actually Parse. After this , any argument could be accessed by args.model.
args = parser.parse_args()

#Error Handling for different arguments.
if args.model not in models:
    raise Exception("Given model string not in valid models. Add your new model to const.py")

#Load data code should be here.

#Call for different models code should be here.

#Dummy code.
print("The model is ", args.model)