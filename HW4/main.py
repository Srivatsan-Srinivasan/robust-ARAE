import argparse
import torch as t
from process_params import check_args, get_params
from const import *
from train_models import train, predict
from data_process import generate_iterators
import ast


t.manual_seed(1)
# Create Parser.
parser = argparse.ArgumentParser(description="For CS287 HW4")


# Add arguments to be parsed.
# GENERAL PARAMS
parser.add_argument('--debug', default=False, type=ast.literal_eval)
parser.add_argument('--cuda', default=CUDA_DEFAULT, type=ast.literal_eval)
parser.add_argument('--exp_n', default='dummy_expt', help='Give name for expt')
parser.add_argument('--monitor', default=False, help='States if you need to pickle validation loss', type=ast.literal_eval)
parser.add_argument('--save', default=False, help='Save the model or not', type=ast.literal_eval)
parser.add_argument('--output_filename', default=None, help='Where the model is saved', type=str)
parser.add_argument('--early_stopping', default=False, help='Whether to stop training once the validation error starts increasing', type=ast.literal_eval)
parser.add_argument('--shuffle', default=True, help='Whether to stop training once the validation error starts increasing', type=ast.literal_eval)

# MODEL PARAMS
parser.add_argument('--model', default='VAE', help='state which model to use')
parser.add_argument('--type', default='MLP', help='what kind of architecture ? For example NLP/CNN/PixelCNN...')  # @todo make it do something in the instantiation of models
parser.add_argument('--n_layers', default=1, type=int)
parser.add_argument('--hidden_dim', default=300, type=int)
parser.add_argument('--latent_dim', default=50, type=int)
parser.add_argument('--batchnorm', default=True, type=ast.literal_eval)  # @todo make it do something in the instantiation of models

# OPTIMIZER PARAMS
parser.add_argument('--optimizer', default='SGD', type=str)
parser.add_argument('--lr_scheduler', default=None, type=str)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--l2_penalty', default=0, type=float)

# TRAIN PARAMS
parser.add_argument('--n_ep', default=30, type=int)
parser.add_argument('--batch_size', default=100, type=int)

# Actually Parse. After this , any argument could be accessed by args.<argument_name>.Also validate.
args = parser.parse_args()
model_params, opt_params, train_params = get_params(args)

# Load data code should be here. Vocab size function of text.

train_iter, val_iter, test_iter = generate_iterators(BATCH_SIZE=args.batch_size, shuffle=args.shuffle)

if False:
    t.backends.cudnn.enabled = True  # False necessary for memory overflows ? It seems not

# Call for different models code should be here.
# Train Model
trained_model = train(args.model,
                      train_iter,
                      val_iter=val_iter,
                      cuda=args.cuda,
                      save=args.save,
                      save_path=args.output_filename,
                      model_params=model_params,
                      early_stopping=args.early_stopping,
                      train_params=train_params,
                      opt_params=opt_params)

# Dummy code.
print("The model is ", args.model)
