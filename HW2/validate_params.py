# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 22:01:14 2018

@author: SrivatsanPC
"""

from const import *
def check_args(args):
    if args.model not in models:
        raise Exception("Given model string not in valid models. Add your new model to const.py")
    if args.optimizer not in optimizers:
        raise Exception("Given optimizer string not in valid models. Add your new model to const.py and train_seqmodels.py")