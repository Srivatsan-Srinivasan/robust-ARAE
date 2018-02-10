# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 22:01:14 2018

@author: SrivatsanPC
"""

from const import *

def get_params(args):
    model_params = {}
    opt_params   = {}
    train_params = {}
    for k,v in model_params_args_map.items():
        model_params[k] = args[v]
    for k,v in opt_params_args_map.items():
        opt_params[k] = args[v]
    for k,v in train_params_args_map.items():
        train_params[k] = [v]
    
    return model_params, opt_params, train_params     

def check_args(args):
    if args.model not in models:
        raise Exception("Given model string not in valid models. Add your new model to const.py")
    if args.optimizer not in optimizers:
        raise Exception("Given optimizer string not in valid models. Add your new model to const.py and train_seqmodels.py")