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
from data_process import generate_iterators

t.manual_seed(1)
# Create Parser.
parser = argparse.ArgumentParser(description="For CS287 HW2")

# Add arguments to be parsed.
# GENERAL PARAMS
parser.add_argument('--debug', default=False)
parser.add_argument('--emb', default='GloVe')
parser.add_argument('--cuda', default=CUDA_DEFAULT)
parser.add_argument('--exp_n', default='dummy_expt', help='Give name for expt')
parser.add_argument('--save', default=False, help='States if you need to pickle validation loss')

# MODEL PARAMS
parser.add_argument('--model', default='NNLM', help='state which model to use')
parser.add_argument('--lstm_nl', default=1)
parser.add_argument('--lstm_h_dim', default=100)
parser.add_argument('--emb_size', default=300)
parser.add_argument('--batch_size', default=10)
parser.add_argument('--dropout', default=0.5)
parser.add_argument('--context_size', default=None)

# OPTIMIZER PARAMS
parser.add_argument('--optimizer', default='SGD')
parser.add_argument('--lr', default=0.1)

# Actually Parse. After this , any argument could be accessed by args.<argument_name>.Also validate.
args = parser.parse_args()
check_args(args)
model_params, opt_params, train_params = get_params(args)

# Load data code should be here. Vocab size function of text.
train_iter, valid_iter, test_iter, TEXT, model_params.vocab_size, embeddings = generate_iterators(debug=args.debug)

# Call for different models code should be here.
# Train Model
trained_model = train(args.model, TEXT.vocab.vectors, train_iter, cuda=args.cuda, context_size=args.context_size,
                      model_params=model_params, train_params=train_params, opt_params=opt_params)

# Predict Model
predict(trained_model, args.model, test_iter, context_size=args.context_size, save_loss=args.save, cuda=args.cuda, expt_name=args.exp_n)

# Dummy code.
print("The model is ", args.model)
