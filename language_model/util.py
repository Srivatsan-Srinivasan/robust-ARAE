import numpy as np
import torch as t
from torchtext.vocab import Vectors, GloVe
import pickle
import os
from sklearn.utils import shuffle
import torchtext
import random

os.chdir('../HW2')  # so that there is not an import bug if the working directory isn't already HW2
from const import *


def length_sort(items, lengths, descending=True):
    """In order to use pytorch variable length sequence package"""
    items = list(zip(items, lengths))
    items.sort(key=lambda x: x[1], reverse=True)
    items, lengths = zip(*items)
    return list(items), list(lengths)


class Corpus(object):
    def __init__(self, path, maxlen, vocab_size=11000, lowercase=False, word2idx=None, idx2word=None):
        """
        :param path: path to the directory containing the dataset
                     ex: snli_lm/
        :param word2idx, idx2word: If you have a saved models, when you do load_models/ae/disc/gen, you have 2 dictionnaries: word2idx and
                                   idx2word.
                                   You can pass them as inputs of the Corpus so that it doesn't recompute the mappings word - idx
                                   Otherwise it would be randomly shuffled
        """
        self.dictionary = Dictionary()
        if word2idx is not None:
            assert idx2word is not None
            self.dictionary.word2idx = word2idx
            self.dictionary.idx2word = idx2word
        self.maxlen = maxlen
        self.lowercase = lowercase
        self.vocab_size = vocab_size
        self.train_path = os.path.join(path, 'train.txt')
        self.test_path = os.path.join(path, 'test.txt')

        # make the vocabulary from training set
        if word2idx is None:
            self.make_vocab()

        self.train = self.tokenize(self.train_path)
        self.test = self.tokenize(self.test_path)

    def make_vocab(self):
        assert os.path.exists(self.train_path)
        # Add words to the dictionary
        with open(self.train_path, 'r') as f:
            for line in f:
                if self.lowercase:
                    # -1 to get rid of \n character
                    words = line[:-1].lower().split(" ")
                else:
                    words = line[:-1].split(" ")
                for word in words:
                    self.dictionary.add_word(word)

        # prune the vocabulary
        self.dictionary.prune_vocab(k=self.vocab_size, cnt=False)

    def tokenize(self, path):
        """Tokenizes a text file."""
        dropped = 0
        with open(path, 'r') as f:
            linecount = 0
            lines = []
            for line in f:
                linecount += 1
                if self.lowercase:
                    words = line[:-1].lower().strip().split(" ")
                else:
                    words = line[:-1].strip().split(" ")
                if len(words) > self.maxlen:
                    dropped += 1
                    continue
                words = ['<sos>'] + words
                words += ['<eos>']
                # vectorize
                vocab = self.dictionary.word2idx
                unk_idx = vocab['<oov>']
                indices = [vocab[w] if w in vocab else unk_idx for w in words]
                lines.append(indices)

        print("Number of sentences dropped from {}: {} out of {} total".
              format(path, dropped, linecount))
        return lines


def batchify(data, bsz, shuffle=False, gpu=False, gpu_id=None):
    """
    Transform a list of data into batched torch Tensors
    :param data: A list of integer-encoded sentences
    :param bsz: batch size
    :param shuffle:
    :param gpu: whether to load the data on gpu
    :return: a list of 3-tuples of the form (source_sentence, FLATTENED_target_sentence, length_of_sentence)
             * All 3 are LongTensor
             * the source and target are the same thing in itself, but the source starts with SOS and the target ends with EOS
             * Note that `FLATTENED_target_sentence` is flattened (batch x n_words,), contrary to `source_sentence`,
               which is (batch, n_words)
    """
    if shuffle:
        random.shuffle(data)
    nbatch = len(data) // bsz
    batches = []

    for i in range(nbatch):
        # Pad batches to maximum sequence length in batch
        batch = data[i * bsz:(i + 1) * bsz]
        # subtract 1 from lengths b/c includes BOTH starts & end symbols
        lengths = [len(x) - 1 for x in batch]
        # sort items by length (decreasing)
        batch, lengths = length_sort(batch, lengths)

        # source has no end symbol
        source = [x[:-1] for x in batch]
        # target has no start symbol
        target = [x[1:] for x in batch]

        # find length to pad to
        maxlen = max(lengths)
        for x, y in zip(source, target):
            zeros = (maxlen - len(x)) * [0]
            x += zeros
            y += zeros

        source = t.LongTensor(np.array(source))
        target = t.LongTensor(np.array(target)).view(-1)

        if gpu:
            source = source.cuda(gpu_id)
            target = target.cuda(gpu_id)

        batches.append((source, target, lengths))

    return batches


def variable(array, requires_grad=False, to_float=True, cuda=CUDA_DEFAULT):
    """Wrapper for t.autograd.Variable"""
    #import pdb; pdb.set_trace()
    if isinstance(array, np.ndarray):
        v = t.autograd.Variable(t.from_numpy(array), requires_grad=requires_grad)
    elif isinstance(array, list) or isinstance(array, tuple):
        v = t.autograd.Variable(t.from_numpy(np.array(array)), requires_grad=requires_grad)
    elif isinstance(array, float) or isinstance(array, int):
        v = t.autograd.Variable(t.from_numpy(np.array([array])), requires_grad=requires_grad)
    elif isinstance(array, t.Tensor) or isinstance(array, t.FloatTensor) or isinstance(array, t.DoubleTensor) or isinstance(array, t.LongTensor) or isinstance(array, t.cuda.FloatTensor) or isinstance(array, t.cuda.DoubleTensor) or isinstance(array, t.cuda.LongTensor):
        v = t.autograd.Variable(array, requires_grad=requires_grad)            
    elif isinstance(array, t.autograd.Variable):
        v = array
    else:
        raise ValueError("type(array): %s" % type(array))
    if cuda and not v.is_cuda:
        v = v.cuda()
    if to_float:
        return v.float()
    else:
        return v


def save_model(model, path):
    t.save(model.state_dict(), path)


def load_model(untrained_model, path, cuda=True):
    if cuda:
        untrained_model.load_state_dict(t.load(path, map_location=lambda storage, loc: storage.cuda(0)))
    else:
        untrained_model.load_state_dict(t.load(path, map_location=lambda storage, loc: storage))


def batch2text(batch, TEXT):
    return " ".join([TEXT.vocab.itos[i] for i in batch.text[:, 0].data])


def pickle_entry(entry, name):
    pickle.dump(entry, open(name + ".p", "wb"))


def load_pickle_entry(file_name):
    return pickle.load(open(file_name, "rb"))


def data_generator(iterator, model_str, context_size=None, cuda=True):
    """
    Treats differently NNLM2 from other models

    :yield: (x,y) pairs
        For NNLM2: x is the context, y is the word that follows
        For the rest: x is the sentence, y is the shiffted sentence
    """
    if model_str != 'NNLM2':
        for i, next_batch in enumerate(iterator):
            if i == 0:
                current_batch = next_batch
            else:
                if model_str == 'NNLM':
                    if context_size is not None:
                        if i > 1:
                            starting_words = last_batch.text.transpose(0, 1)[:, -context_size:]
                        else:
                            starting_words = t.zeros(current_batch.text.size(1), context_size).float()
                        x = t.cat([variable(starting_words, to_float=False, cuda=cuda).long(), variable(current_batch.text.transpose(0, 1).data, cuda=cuda, to_float=False).long()], 1)
                    else:
                        raise ValueError('`context_size` should not be None for NNLM')
                else:
                    x = variable(current_batch.text.transpose(0, 1).data, to_float=False, cuda=cuda).long()

                if model_str == 'NNLM':
                    # for CNN, you predict all the time steps between 0 and T-1 included
                    # you do not predict the time step T (time step 0 of next batch)
                    target = variable(current_batch.text.transpose(0, 1).data, to_float=False, cuda=cuda)
                elif model_str in recur_models:
                    # for RNN, you predict all the time steps between 1 and T-1, as well as T (0th of the next batch)
                    target = t.cat([variable(current_batch.text.transpose(0, 1)[:, 1:].data, to_float=False, cuda=cuda).long(),
                                    variable(next_batch.text.transpose(0, 1)[:, :1].data, cuda=cuda, to_float=False).long()],
                                   1)
                else:
                    raise NotImplementedError("Not implemented or not put into the right list in const.py")

                last_batch = current_batch
                current_batch = next_batch

                yield x, target
    else:
        """In that case `iterator` is  a `namedtuple` with fields `dataset`and `batch_size`"""
        batch_size = iterator.batch_size
        dataset_ = shuffle(iterator.dataset)
        for k in range(0, len(dataset_), batch_size):
            batch = np.concatenate(dataset_[k:k + batch_size], 0)
            x = variable(batch[:, :-1], to_float=False, cuda=cuda).long()
            y = variable(batch[:, -1], to_float=False, cuda=cuda).long()
            yield x, y


# UTILS FOR THE CNN-IMPLEMENTED NNLM
# SHUFFLE THE TRAINING TEXT FILE AND RECREATE THE ITERATOR (IID ASSUMPTION)
def shuffle_train_txt_file(input_filename, output_filename):
    with open(input_filename, 'r') as ifile:
        text = shuffle(ifile.read().split('\n'))
    with open(output_filename, 'w') as ofile:
        ofile.write('\n'.join(text))


def rebuild_iterators(TEXT, batch_size=10):
    """
    Shuffle the train.txt file and recreate the iterators
    :param TEXT:
    :return:
    """
    if 'shuffled_train.txt' in os.listdir():
        shuffle_train_txt_file('shuffled_train.txt', 'shuffled_train.txt')
    else:
        shuffle_train_txt_file('train.txt', 'shuffled_train.txt')
    train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
        path='../HW2/',
        train="shuffled_train.txt", validation="valid.txt", test="valid.txt", text_field=TEXT)
    train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
        (train, val, test), batch_size=batch_size, device=-1, bptt_len=32, repeat=False, shuffle=False)
    return train_iter, val_iter, test_iter


class ReduceLROnPlateau(object):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. Default: 10.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = ReduceLROnPlateau(optimizer, 'min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     # Note that step should be called after validate()
        >>>     scheduler.step(val_loss)
    """

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        if not isinstance(optimizer, t.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.eps = eps
        self.last_epoch = -1
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print('Epoch {:5d}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            self.is_better = lambda a, best: a < best * rel_epsilon
            self.mode_worse = float('Inf')
        elif mode == 'min' and threshold_mode == 'abs':
            self.is_better = lambda a, best: a < best - threshold
            self.mode_worse = float('Inf')
        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            self.is_better = lambda a, best: a > best * rel_epsilon
            self.mode_worse = -float('Inf')
        else:  # mode == 'max' and epsilon_mode == 'abs':
            self.is_better = lambda a, best: a > best + threshold

            self.mode_worse = -float('Inf')


class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, t.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class LambdaLR(_LRScheduler):
    """Sets the learning rate of each parameter group to the initial lr
    times a given function. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in optimizer.param_groups.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # Assuming optimizer has two groups.
        >>> lambda1 = lambda epoch: epoch // 30
        >>> lambda2 = lambda epoch: 0.95 ** epoch
        >>> scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
        >>> for epoch in range(100):
        >>>     scheduler.step()
        >>>     train(...)
        >>>     validate(...)
    """
    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        self.optimizer = optimizer
        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError("Expected {} lr_lambdas, but got {}".format(
                    len(optimizer.param_groups), len(lr_lambda)))
            self.lr_lambdas = list(lr_lambda)
        self.last_epoch = last_epoch
        super(LambdaLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * lmbda(self.last_epoch)

                for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]