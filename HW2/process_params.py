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
    args = vars(args)
    for k,v in model_params_args_map.items():
        if v in args:
            model_params[k] = args[v]
    for k,v in opt_params_args_map.items():
        if v in args:
            opt_params[k] = args[v]
    for k,v in train_params_args_map.items():
        if v in args:
            train_params[k] = args[v]
   
    return model_params, opt_params, train_params     

def check_args(args):
    if args.model not in models:
        raise Exception("Given model string not in valid models. Add your new model to const.py"+ 
                        "Could also be case mismatch. Check const.py")
    if args.optimizer not in optimizers:
        raise Exception("Given optimizer string not in valid models. Add your new model to const.py and train_seqmodels.py" +
                        "Could also be case mismatch. Check const.py")
    if args.emb not in embeddings:            
        raise Exception("Given embedding string not in valid models. Add your new embedding to const.py and data_process.py" +
                        "Could also be case mismatch. Check const.py")