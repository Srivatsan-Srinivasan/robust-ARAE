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
from train_models import train, predict
from data_process import generate_iterators, generate_text
from utils import EOS_WORD, BOS_WORD, tokenize_de, tokenize_en
t.manual_seed(1)
# Create Parser.
parser = argparse.ArgumentParser(description="For CS287 HW3")

# Add arguments to be parsed.
# GENERAL PARAMS
parser.add_argument('--debug', default=False, type=bool)
parser.add_argument('--cuda', default=CUDA_DEFAULT, type=bool)
parser.add_argument('--exp_n', default='dummy_expt', help='Give name for expt')
parser.add_argument('--monitor', default=False, help='States if you need to pickle validation loss', type=bool)
parser.add_argument('--save', default=False, help='Save the model or not', type=bool)
parser.add_argument('--output_filename', default=None, help='Where the model is saved', type=str)
parser.add_argument('--early_stopping', default=False, help='Whether to stop training once the validation error starts increasing', type=bool)
parser.add_argument('--max_len', default=20, type=int)
parser.add_argument('--load_saved_data', default=False, type=bool)

# MODEL PARAMS
parser.add_argument('--model', default='LSTM', help='state which model to use')
parser.add_argument('--n_layers', default=1, type=int)
parser.add_argument('--hidden_dim', default=300, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument('--embed_dropout', default=False, type=bool)
parser.add_argument('--emb_train', default=True, type=bool)
parser.add_argument('--clip_gradients', default=5., type=float)
parser.add_argument('--embedding_dim', default=50, type=int)

# OPTIMIZER PARAMS
parser.add_argument('--optimizer', default='SGD', type=str)
parser.add_argument('--lr_scheduler', default=None, type=str)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--l2_penalty', default=0, type=float)

# TRAIN PARAMS
parser.add_argument('--n_ep', default=30, type=int)

# Actually Parse. After this , any argument could be accessed by args.<argument_name>.Also validate.
args = parser.parse_args()
check_args(args)
model_params, opt_params, train_params = get_params(args)

# Load data code should be here. Vocab size function of text.
train_iter, val_iter, EN, DE = generate_iterators(MAX_LEN=args.max_len, load_data=args.load_saved_data, BATCH_SIZE=args.batch_size)
model_params['source_vocab_size'] = len(DE.vocab.itos)
model_params['target_vocab_size'] = len(EN.vocab.itos)

if False:  # necessary for memory overflows ?
    t.backends.cudnn.enabled = False

# Call for different models code should be here.
# Train Model
trained_model = train(args.model,
                      train_iter,
                      val_iter=val_iter,
                      cuda=args.cuda,
                      save=args.save,
                      save_path=args.output_filename,
                      source_embedding=None,
                      target_embedding=None,
                      model_params=model_params,
                      early_stopping=args.early_stopping,
                      train_params=train_params,
                      opt_params=opt_params)


# @todo : implement this
# generate_text(trained_model, args.exp_n, TEXT, n=20, cuda=args.cuda)

# Dummy code.
print("The model is ", args.model)
