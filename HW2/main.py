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

t.manual_seed(1)
# Create Parser.
parser = argparse.ArgumentParser(description="For CS287 HW2")

# Add arguments to be parsed.
# GENERAL PARAMS
parser.add_argument('--debug', default=False, type=bool)
parser.add_argument('--emb', default='GloVe')
parser.add_argument('--cuda', default=CUDA_DEFAULT, type=bool)
parser.add_argument('--exp_n', default='dummy_expt', help='Give name for expt')
parser.add_argument('--save', default=False, help='States if you need to pickle validation loss', type=bool)

# MODEL PARAMS
parser.add_argument('--model', default='NNLM', help='state which model to use')
parser.add_argument('--lstm_nl', default=1, type=int)
parser.add_argument('--lstm_h_dim', default=100, type=int)
parser.add_argument('--emb_size', default=300, type=int)
parser.add_argument('--batch_size', default=10, type=int)
parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument('--con_size', default=-1, type=int)
parser.add_argument('--emb_train', default=False, type=bool)
parser.add_argument('--clip_g_n', default=0.25, type=float)

# OPTIMIZER PARAMS
parser.add_argument('--optimizer', default='SGD')
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--l2_penalty', default=0, type=float)

# TRAIN PARAMS
parser.add_argument('--n_ep', default=30, type=int)

# Actually Parse. After this , any argument could be accessed by args.<argument_name>.Also validate.
args = parser.parse_args()
check_args(args)
model_params, opt_params, train_params = get_params(args)

# Load data code should be here. Vocab size function of text.
train_iter, valid_iter, test_iter, TEXT, model_params['vocab_size'], embeddings = generate_iterators(args.model, debug=args.debug, batch_size=args.batch_size, context_size=model_params['context_size'])

# Call for different models code should be here.
# Train Model
trained_model = train(args.model, TEXT.vocab.vectors, train_iter, val_iter=valid_iter, cuda=args.cuda,
                      context_size=int(args.con_size), model_params=model_params,
                      train_params=train_params, opt_params=opt_params, TEXT=TEXT, reshuffle_train=(args.model == 'NNLM'))

# Predict Model
# @todo: make it work for nnlm2
predict(trained_model, args.model, test_iter, context_size=int(args.con_size),
        save_loss=args.save, cuda=args.cuda, expt_name=args.exp_n)

generate_text(trained_model, args.exp_n, TEXT, n=20, cuda=args.cuda)

# Dummy code.
print("The model is ", args.model)
