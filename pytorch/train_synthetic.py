import argparse
import json
import math
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import Seq2Seq, MLP_D, MLP_G
from train_utils import save_model, evaluate_autoencoder_synthetic, train_lm_synthetic, train_ae, train_gan_g, train_gan_d, get_optimizers_gan, get_synthetic_dataset, load_oracle
from utils import to_gpu, SyntheticCorpus, batchify, activation_from_str, tensorboard, create_tensorboard_dir, check_args, Timer
import ast


# Terminal arg parsing
def init_config():
    """Just to make it more readable in PyCharm"""
    parser = argparse.ArgumentParser(description='PyTorch ARAE for Text')

    def path(parser):
        # Path Arguments
        parser.add_argument('--data_path', type=str, required=True,
                            help='location of the data corpus')
        parser.add_argument('--kenlm_path', type=str, default='../Data/kenlm',
                            help='path to kenlm directory')
        parser.add_argument('--outf', type=str, required=True,
                            help='output directory name')
        return parser

    def preprocessing(parser):
        # Data Processing Arguments
        parser.add_argument('--vocab_size', type=int, default=11000,
                            help='cut vocabulary down to this size '
                                 '(most frequently seen words in train)')
        parser.add_argument('--maxlen', type=int, default=30,
                            help='maximum sentence length')
        parser.add_argument('--lowercase', action='store_true',
                            help='lowercase all text')
        return parser

    def model(parser):
        # Model Arguments
        parser.add_argument('--emsize', type=int, default=300,
                            help='size of word embeddings')
        parser.add_argument('--nhidden_enc', type=int, default=300,
                            help='number of hidden units per layer')
        parser.add_argument('--nhidden_dec', type=int, default=300,
                            help='number of hidden units per layer')
        parser.add_argument('--nlayers', type=int, default=1,
                            help='number of layers')
        parser.add_argument('--noise_radius', type=float, default=0.2,
                            help='stdev of noise for autoencoder (regularizer)')
        parser.add_argument('--noise_anneal', type=float, default=0.995,
                            help='anneal noise_radius exponentially by this'
                                 'every 100 iterations')
        parser.add_argument('--hidden_init', action='store_true',
                            help="initialize decoder hidden state with encoder's")
        parser.add_argument('--arch_g', type=str, default='300-300',
                            help='generator architecture (MLP)')
        parser.add_argument('--arch_d', type=str, default='300-300',
                            help='critic/discriminator architecture (MLP)')
        parser.add_argument('--z_size', type=int, default=100,
                            help='dimension of random noise z to feed into generator')
        parser.add_argument('--temp', type=float, default=1,
                            help='softmax temperature (lower --> more discrete)')
        parser.add_argument('--enc_grad_norm', type=ast.literal_eval, default=True,
                            help='norm code gradient from critic->encoder')
        parser.add_argument('--gan_toenc', type=float, default=-0.01,
                            help='weight factor passing gradient from gan to encoder')
        parser.add_argument('--dropout', type=float, default=0.0,
                            help='dropout applied to layers (0 = no dropout)')
        parser.add_argument('--gan_weight_init', type=str, default='default',  # @todo: compare He initalization with default initialization
                            help='What initializer you would like to use. `default` (Yoon\'s version) or `he` suppported')
        parser.add_argument('--gan_activation', default='lrelu', type=str,
                            help='Activation to use in GAN')
        parser.add_argument('--std_minibatch', action='store_true',
                            help="Whether to compute minibatch std in the discriminator as an additional feature")
        parser.add_argument('--bn_disc', action='store_true',
                            help="Whether to use batchnorm in the discriminator")
        parser.add_argument('--bn_gen', action='store_true',
                            help="Whether to use batchnorm in the generator")
        parser.add_argument('--l2_reg_disc', type=float, default=None,
                            help="Whether to use l2 regularization on the last layer of the discriminator (it tends to diverge)"
                                 "Try with 100 = 10^2 = 1/sig^2")
        parser.add_argument('--tie_weights', action='store_true',
                            help="Whether to tie the weights of the embedding of the decoder and its linear layer")
        parser.add_argument('--bidirectionnal', action='store_true',
                            help="Whether the encoder should be bidirectionnal. If it is, it divides the hdim of the encoder by 2")
        parser.add_argument('--polar', action='store_true',
                            help='Whether to use polar interpolation for GP')
        parser.add_argument('--norm_penalty', type=float, default=None,
                            help='If you want to enforce a L2 regularization on the norm of the encoder, instead of forcing it to lie'
                                 'on the unit sphere. This way you can use volume (as you do interpolation it may make more sense)'
                                 'If you try it, use 1-10 as a starting point')
        parser.add_argument('--norm_penalty_threshold', type=float, default=0.,
                            help='Whether you want to penalize the norm of the code for being above this value')
        return parser

    def training(parser):
        # Training Arguments
        parser.add_argument('--epochs', type=int, default=15,
                            help='maximum number of epochs')
        parser.add_argument('--min_epochs', type=int, default=6,
                            help="minimum number of epochs to train for")
        parser.add_argument('--no_earlystopping', action='store_true',
                            help="won't use KenLM for early stopping")
        parser.add_argument('--patience', type=int, default=5,
                            help="number of language model evaluations without ppl "
                                 "improvement to wait before early stopping")
        parser.add_argument('--ae_lr_scheduler', action='store_true',
                            help='Whether to use a scheduler to decrease AE learning rate when PPL doesnt decrease')
        parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                            help='batch size')
        parser.add_argument('--niters_ae', type=int, default=1,
                            help='number of autoencoder iterations in training')
        parser.add_argument('--niters_gan_d', type=int, default=5,
                            help='number of discriminator iterations in training')
        parser.add_argument('--niters_gan_g', type=int, default=1,
                            help='number of generator iterations in training')
        parser.add_argument('--niters_gan_schedule', type=str, default='2-4-6',
                            help='epoch counts to increase number of GAN training '
                                 ' iterations (increment by 1 each time)')
        parser.add_argument('--optim_gan', default='adam', type=str,
                            help='rmsprop or adam')
        parser.add_argument('--lr_ae', type=float, default=1,
                            help='autoencoder learning rate')
        parser.add_argument('--lr_gan_g', type=float, default=5e-05,
                            help='generator learning rate')
        parser.add_argument('--lr_gan_d', type=float, default=1e-05,
                            help='critic/discriminator learning rate')
        parser.add_argument('--beta1', type=float, default=0.9,
                            help='beta1 for adam. default=0.9')
        parser.add_argument('--clip', type=float, default=1,
                            help='gradient clipping, max norm')
        parser.add_argument('--gan_clamp', type=float, default=0.01,
                            help='WGAN clamp')
        parser.add_argument('--gradient_penalty', action='store_true',
                            help='Whether to use a gradient penalty in the discriminator loss, instead of the weight clipping')
        parser.add_argument('--lambda_GP', type=float, default=10.,
                            help='Regularization param for the gradient penalty')
        parser.add_argument('--spectralnorm', action='store_true',
                            help='Whether to use a spectral normalization in the discriminator loss')
        parser.add_argument('--lambda_dropout', type=float, default=None,
                            help='The coefficient in front of the dropout penalty'
                                 '2 is the value they use in the paper')
        parser.add_argument('--dropout_penalty', type=float, default=None,
                            help='To enforce the Lipschitz continuity of the critic'
                                 'See paper `Improving the improved WGAN`'
                                 'Should be a small value (try .05)'
                                 'Additional loss added to the critic')
        parser.add_argument('--progressive_vocab', action='store_true',
                            help='Whether to train sequentially with increasing vocab')
        parser.add_argument('--eps_drift', type=float, default=None,
                            help='Whether to add a term eps_drift*D(x)^2 in the loss of the discriminator'
                                 'If None, add nothing')
        return parser

    def eval(parser):
        # Evaluation Arguments
        parser.add_argument('--sample', action='store_true',
                            help='sample when decoding for generation')
        parser.add_argument('--N', type=int, default=5,
                            help='N-gram order for training n-gram language model')
        parser.add_argument('--log_interval', type=int, default=200,
                            help='interval to log autoencoder training results')
        return parser

    def other(parser):
        # Other
        parser.add_argument('--seed', type=int, default=1111,
                            help='random seed')
        parser.add_argument('--cuda', action='store_true',
                            help='use CUDA')
        parser.add_argument('--n_gpus', type=int, default=1,  # @todo : no speedup even though GPUs are used (see https://discuss.pytorch.org/t/debugging-dataparallel-no-speedup-and-uneven-memory-allocation/1100/14)
                            help='The number of GPUs you want to use')
        parser.add_argument('--gpu_id', type=int, default=None,
                            help='If you have several GPUs but want to use one in particular, specify the ID here')
        parser.add_argument('--tensorboard', action='store_true',
                            help='Whether to use tensorboard or not')
        parser.add_argument('--tensorboard_freq', type=int, default=300,
                            help='logging frequency')
        parser.add_argument('--tensorboard_logdir', type=str, default=None,  # by default tensorboard/ (just add the relative path from tensorboard/)
                            help='Tensorboard logging directory. It will be a subdirectory of `tensorboard/`, so don\'t had the prefix before your name!')
        parser.add_argument('--timeit', type=int, default=None,
                            help='Whether to time functions or not. If you indicate nothing, it is None and nothing happens'
                                 'Otherwise, you should indicate a log frequency. Don\'t make it too small or tensorboard will overflow'
                                 'Try 5000 (it will time functions 1/5000 of the time)')
        parser.add_argument('--save_intermediate', type=float, default=None,
                            help='If not None, save also the models that have a reverse ppl less than best+save_intermediate')

        return parser

    parser = other(eval(training(model(preprocessing(path(parser))))))

    args = parser.parse_args()
    return args


args = init_config()
print(vars(args))
check_args(args)

# make output directory if it doesn't already exist
if not os.path.isdir('./output'):
    os.makedirs('./output')
if not os.path.isdir('./output/{}'.format(args.outf)):
    os.makedirs('./output/{}'.format(args.outf))

# Set the random seed manually for reproducibility.
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, "
              "so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

# create corpus

train, test = get_synthetic_dataset(args)
corpus = SyntheticCorpus(train, test,
                         maxlen=args.maxlen,
                         vocab_size=args.vocab_size)

# save arguments
ntokens = corpus.vocab_size
print("Vocabulary Size: {}".format(ntokens))
args.ntokens = ntokens
oracle = load_oracle(args)
with open('./output/{}/args.json'.format(args.outf), 'w') as f:
    json.dump(vars(args), f)
with open("./output/{}/logs.txt".format(args.outf), 'w') as f:
    f.write(str(vars(args)))
    f.write("\n\n")

eval_batch_size = 10

print("Loaded data!")

###############################################################################
# Build the models
###############################################################################

ntokens = corpus.vocab_size + 3
create_tensorboard_dir(args.tensorboard_logdir) if args.tensorboard else None
writer = SummaryWriter(log_dir='tensorboard/' + args.tensorboard_logdir) if args.tensorboard else None
global_timer = Timer('global', enabled=args.timeit is None, log_freq=args.timeit, writer=writer)  # @todo: time train functions with this one

autoencoder = Seq2Seq(emsize=args.emsize,
                      nhidden_enc=args.nhidden_enc,
                      nhidden_dec=args.nhidden_dec,
                      ntokens=ntokens,
                      nlayers=args.nlayers,
                      noise_radius=args.noise_radius,
                      hidden_init=args.hidden_init,
                      dropout=args.dropout,
                      gpu=args.cuda,
                      ngpus=args.n_gpus,
                      gpu_id=args.gpu_id,
                      timeit=args.timeit,
                      writer=writer,
                      tie_weights=args.tie_weights,
                      norm_penalty=args.norm_penalty,
                      norm_penalty_threshold=args.norm_penalty_threshold,
                      bidirectionnal=args.bidirectionnal
                      )
gan_gen = MLP_G(ninput=args.z_size, noutput=args.nhidden_enc, layers=args.arch_g, activation=activation_from_str(args.gan_activation),
                weight_init=args.gan_weight_init, batchnorm=args.bn_gen, gpu=args.cuda, gpu_id=args.gpu_id, timeit=args.timeit, writer=writer)
gan_disc = MLP_D(ninput=args.nhidden_enc, noutput=1, layers=args.arch_d, activation=activation_from_str(args.gan_activation),
                 weight_init=args.gan_weight_init, std_minibatch=args.std_minibatch, batchnorm=args.bn_disc, polar=args.polar,
                 spectralnorm=args.spectralnorm, gpu=args.cuda, writer=writer, gpu_id=args.gpu_id, lambda_GP=args.lambda_GP,
                 timeit=args.timeit, lambda_dropout=args.lambda_dropout, dropout=args.dropout_penalty)

criterion_ce = nn.CrossEntropyLoss()

if args.cuda:
    autoencoder = autoencoder.cuda(args.gpu_id)
    gan_gen = gan_gen.cuda(args.gpu_id)
    gan_disc = gan_disc.cuda(args.gpu_id)
    criterion_ce = criterion_ce.cuda(args.gpu_id)

if torch.cuda.device_count() > 1 and args.n_gpus > 1:
    print("Let's use", args.n_gpus, "GPUs!")
    gan_gen = nn.DataParallel(gan_gen)
    gan_disc = nn.DataParallel(gan_disc)
    autoencoder = nn.DataParallel(autoencoder)

print(autoencoder)
print(gan_gen)
print(gan_disc)

optimizer_ae = optim.SGD(autoencoder.parameters(), lr=args.lr_ae)
optimizer_gan_g, optimizer_gan_d = get_optimizers_gan(gan_gen, gan_disc, args)

scheduler = None
if args.ae_lr_scheduler:
    scheduler = ReduceLROnPlateau(optimizer_ae, mode='min', factor=.5, patience=1, threshold=1e-3)

# This will still retain overall number of tokens to the initial vocabulary size, just modify data.
test_data = batchify(corpus.test, eval_batch_size, shuffle=False, gpu_id=args.gpu_id)
train_data = batchify(corpus.train, args.batch_size, shuffle=True, gpu_id=args.gpu_id)
print('Train data has %d batches' % len(train_data))

###############################################################################
# Training code
###############################################################################
print("Training...")
with open("./output/{}/logs.txt".format(args.outf), 'a') as f:
    f.write('Training...\n')

# schedule of increasing GAN training loops
if args.niters_gan_schedule != "":
    gan_schedule = [int(x) for x in args.niters_gan_schedule.split("-")]
else:
    gan_schedule = []
niter_gan = 1

fixed_noise = to_gpu(args.cuda,
                     Variable(torch.ones(args.batch_size, args.z_size)),
                     args.gpu_id)
fixed_noise.data.normal_(0, 1)

best_ppl = None
impatience = 0
all_ppl = []
for epoch in range(1, args.epochs + 1):
    # update gan training schedule
    if epoch in gan_schedule:
        niter_gan += 1
        print("GAN training loop schedule increased to {}".format(niter_gan))
        with open("./output/{}/logs.txt".format(args.outf), 'a') as f:
            f.write("GAN training loop schedule increased to {}\n".
                    format(niter_gan))

    total_loss_ae = 0
    epoch_start_time = time.time()
    start_time = time.time()
    niter = 0
    niter_global = 1

    # loop through all batches in training data
    while niter < len(train_data):
        if impatience > args.patience: break

        # train autoencoder ----------------------------
        for i in range(args.niters_ae):
            if niter == len(train_data):
                break  # end of
            total_loss_ae, start_time = train_ae(autoencoder, criterion_ce, optimizer_ae, train_data, train_data[niter], total_loss_ae, start_time, i, ntokens, epoch, args, writer, niter_global + (-1 + epoch) * len(train_data))
            niter += 1

        # train gan ----------------------------------
        for k in range(niter_gan):

            # train discriminator/critic
            for i in range(args.niters_gan_d):
                # feed a seen sample within this epoch; good for early training
                errD, errD_real, errD_fake = train_gan_d(autoencoder, gan_disc, gan_gen, optimizer_gan_d, optimizer_ae, train_data[random.randint(0, len(train_data) - 1)], args, writer, niter_global + (-1 + epoch) * len(train_data))

            # train generator
            for i in range(args.niters_gan_g):
                errG = train_gan_g(gan_gen, gan_disc, optimizer_gan_g, args, writer, niter_global + (-1 + epoch) * len(train_data))

        niter_global += 1

        tensorboard(niter_global + (-1 + epoch) * len(train_data), writer, gan_gen, gan_disc, autoencoder, args.tensorboard_freq) if args.n_gpus == 1 else tensorboard(niter_global, writer, gan_gen.module, gan_disc.module, autoencoder.module,
                                                                                                                                                                       args.tensorboard_freq)
        if niter_global % 100 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.8f (Loss_D_real: %.8f '
                  'Loss_D_fake: %.8f) Loss_G: %.8f'
                  % (epoch, args.epochs, niter, len(train_data),
                     errD.data[0], errD_real.data[0],
                     errD_fake.data[0], errG.data[0]))
            with open("./output/{}/logs.txt".format(args.outf), 'a') as f:
                f.write('[%d/%d][%d/%d] Loss_D: %.8f (Loss_D_real: %.8f '
                        'Loss_D_fake: %.8f) Loss_G: %.8f\n'
                        % (epoch, args.epochs, niter, len(train_data),
                           errD.data[0], errD_real.data[0],
                           errD_fake.data[0], errG.data[0]))

            # exponentially decaying noise on autoencoder
            if args.n_gpus == 1:
                autoencoder.noise_radius = autoencoder.noise_radius * args.noise_anneal
            else:  # in that case `autoencoder` is a DataParallel object, and we have to access its module
                autoencoder.module.noise_radius = autoencoder.module.noise_radius * args.noise_anneal

            if niter_global % 3000 == 0:
                ppl = train_lm_synthetic(gan_gen, autoencoder, oracle, args)

                # evaluate with lm
                if not args.no_earlystopping and epoch > args.min_epochs:
                    if best_ppl is None or ppl < best_ppl:
                        impatience = 0
                        best_ppl = ppl
                        print("New best ppl {}\n".format(best_ppl))
                        with open("./output/{}/logs.txt".format(args.outf), 'a') as f:
                            f.write("New best ppl {}\n".format(best_ppl))
                        save_model(autoencoder, gan_gen, gan_disc, args)
                    else:
                        if args.save_intermediate is not None:
                            if best_ppl + args.save_intermediate >= ppl:
                                save_model(autoencoder, gan_gen, gan_disc, args, intermediate=True, ppl=ppl)

                        impatience += 1
                        # end training
                        if impatience > args.patience:
                            print("Ending training")
                            with open("./output/{}/logs.txt".
                                              format(args.outf), 'a') as f:
                                f.write("\nEnding Training\n")

    # end of epoch ----------------------------
    # evaluation
    test_loss, accuracy = evaluate_autoencoder_synthetic(autoencoder, corpus, criterion_ce, test_data, epoch, args)
    if args.tensorboard:
        writer.add_scalar('acc_recons', accuracy, niter_global + (epoch - 1) * len(train_data))
        writer.add_scalar('test_recons_loss', test_loss, niter_global + (epoch - 1) * len(train_data))

    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
          'test ppl {:5.2f} | acc {:3.3f}'.
          format(epoch, (time.time() - epoch_start_time),
                 test_loss, math.exp(test_loss), accuracy))
    print('-' * 89)

    with open("./output/{}/logs.txt".format(args.outf), 'a') as f:
        f.write('-' * 89)
        f.write('\n| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} |'
                ' test ppl {:5.2f} | acc {:3.3f}\n'.
                format(epoch, (time.time() - epoch_start_time),
                       test_loss, math.exp(test_loss), accuracy))
        f.write('-' * 89)
        f.write('\n')

    if not args.no_earlystopping and epoch >= args.min_epochs:
        eval_path = os.path.join(args.data_path, "test.txt")
        save_path = "./output/{}/end_of_epoch{}_lm_generations".format(args.outf, epoch)
        ppl = train_lm_synthetic(gan_gen, autoencoder, oracle, args)
        scheduler.step(ppl) if scheduler is not None else None
        if args.tensorboard:
            writer.add_scalar('ppl', ppl, niter_global + (epoch - 1) * len(train_data))

        print("Perplexity {}".format(ppl))
        all_ppl.append(ppl)
        print(all_ppl)
        with open("./output/{}/logs.txt".format(args.outf), 'a') as f:
            f.write("\n\nPerplexity {}\n".format(ppl))
            f.write(str(all_ppl) + "\n\n")
        if best_ppl is None or ppl < best_ppl:
            impatience = 0
            best_ppl = ppl
            print("New best ppl {}\n".format(best_ppl))
            with open("./output/{}/logs.txt".format(args.outf), 'a') as f:
                f.write("New best ppl {}\n".format(best_ppl))
            save_model(autoencoder, gan_gen, gan_disc, args)
        else:
            if args.save_intermediate is not None:
                if best_ppl + args.save_intermediate >= ppl:
                    save_model(autoencoder, gan_gen, gan_disc, args, intermediate=True, ppl=ppl)

            impatience += 1
            # end training
            if impatience > args.patience:
                print("Ending training")
                with open("./output/{}/logs.txt".format(args.outf), 'a') as f:
                    f.write("\nEnding Training\n")
                sys.exit()

    # shuffle between epochs
    train_data = batchify(corpus.train, args.batch_size, shuffle=True, gpu_id=args.gpu_id)
