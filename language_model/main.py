# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 15:25:48 2018

@author: SrivatsanPC
"""

# Mostly dummy code for now to illustrate argparser.
import argparse
import torch as t
from process_params import check_args, get_params
from const import *
from train_seqmodels import train, predict
from data_process import generate_iterators, generate_text
import ast

t.manual_seed(1)
# Create Parser.
parser = argparse.ArgumentParser(description="For CS287 HW2")

# Add arguments to be parsed.
# GENERAL PARAMS
parser.add_argument('--debug', default=False, type=ast.literal_eval)
parser.add_argument('--emb', default=None)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--exp_n', default='dummy_expt', help='Give name for expt')
parser.add_argument('--monitor', default=False, help='States if you need to pickle validation loss', type=ast.literal_eval)
parser.add_argument('--save', action='store_true')
parser.add_argument('--output_filename', default=None, help='Where the model is saved', type=str)
parser.add_argument('--data_path', required=True, type=str)
parser.add_argument('--early_stopping', action='store_true')

# MODEL PARAMS
parser.add_argument('--model', default='LSTM', help='state which model to use')
parser.add_argument('--vocab_size', default=11000, type=int, help='state which model to use')
parser.add_argument('--lstm_nl', default=1, type=int)
parser.add_argument('--lstm_h_dim', default=300, type=int)
parser.add_argument('--emb_size', default=300, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument('--embed_dropout', action='store_true')
parser.add_argument('--emb_train', default=True, type=ast.literal_eval)
parser.add_argument('--clip_g_n', default=0.25, type=float)
parser.add_argument('--batch_norm', default=False, type=ast.literal_eval, help='Whether to include batch normalization or not')
parser.add_argument('--tie_weights', default=False, type=ast.literal_eval, help='For LSTM model whether to make output and embedding weights match')
parser.add_argument('--maxlen', default=30, type=int)

# OPTIMIZER PARAMS
parser.add_argument('--optimizer', default='SGD', type=str)
parser.add_argument('--lr_scheduler', default=None, type=str)
parser.add_argument('--lr', default=1., type=float)
parser.add_argument('--l2_penalty', default=0, type=float)

# TRAIN PARAMS
parser.add_argument('--n_ep', default=30, type=int)
parser.add_argument('--gpu_id', default=None, type=int)

# Actually Parse. After this , any argument could be accessed by args.<argument_name>.Also validate.
args = parser.parse_args()
check_args(args)
model_params, opt_params, train_params = get_params(args)

# Load data code should be here. Vocab size function of text.
train_iter, test_iter, corpus, ntokens = generate_iterators(args)

# Call for different models code should be here.
# Train Model
trained_model = train(train_iter, corpus, ntokens, val_iter=test_iter, early_stopping=args.early_stopping, save=args.save, save_path=args.output_filename,
                      model_params=model_params, opt_params=opt_params, train_params=train_params, cuda=args.cuda, gpu_id=args.gpu_id)


# Dummy code.
print("The model is ", args.model)
