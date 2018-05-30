import os
import numpy as np
import random
import torch as t
import time
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence


class Timer(object):
    """
    EXAMPLE:

    timer = Timer()


    @timer.timeit
    def get_all_employee_details(**kwargs):
        print('employee details')

        return 6, 7, 8


    employees = get_all_employee_details(x=5)
    print(employees)
    """

    def __init__(self, name, enabled=False, log_freq=1000, writer=None):
        self.name = name  # to avoid that method with the same name get the same curve (ex: forwards of different classes)
        self.log_freq = log_freq  # to avoid spamming tensorboard too much
        self.writer = writer
        self.enabled = enabled
        self.method_counter = {}

    def timeit(self, method):
        if self.enabled:

            def timed(*args, **kw):

                ts = time.time()
                result = method(*args, **kw)
                te = time.time()

                if method.__name__ in self.method_counter:
                    self.method_counter[method.__name__] += 1
                else:
                    self.method_counter[method.__name__] = 0
                if self.method_counter[method.__name__] % self.log_freq == 0:
                    self.writer.add_scalar(self.name + '_' + method.__name__ + '_timer', te - ts, self.method_counter[method.__name__])
                return result
        else:
            timed = method

        return timed


def load_kenlm():
    global kenlm
    import kenlm


def check_args(args):
    if args.gradient_penalty and args.spectralnorm:
        raise ValueError("You cannot have spectral normalization AND gradient penalty at the same time")
    if args.tensorboard and args.tensorboard_logdir is None:
        raise ValueError("You should provide a name for tensorboard_logdir. Note that you do not need to "
                         "indicate `/tensorboard` as a prefix, the it will automatically be a subfolder of `/tensorboard`")
    if args.gpu_id is not None:
        assert 0 <= args.gpu_id <= t.cuda.device_count() - 1
    if args.gpu_id is not None and args.n_gpus > 1:
        raise ValueError("If you decide to use a specific GPU (args.gpu_id is not None), you cannot also choose to use all of them (args.n_gpus > 1)")
    assert isinstance(args.norm_penalty_threshold, float)
    if args.norm_penalty_threshold != 0:
        assert args.norm_penalty_threshold > 0


def create_tensorboard_dir(logdir):
    if logdir not in os.listdir('tensorboard/'):
        os.makedirs('tensorboard/' + logdir)


# @todo: improve this so that it can efficiently process batch of sentences of different lengths (it should not swap padding with characters)
def noisy_sentence(sentence, k):
    """
    Swap word in a sentence
    :param sentence: a variable containing a LongTensor with dim `length`
    :param k: number of swaps to operate
    :return: swapped sentence
    """
    noisy_s = sentence.data
    for _ in range(k):
        i, j = np.random.randint(0, sentence.size(0), size=2)
        tmp = noisy_s[i] * 1.
        noisy_s[i] = noisy_s[j] * 1.
        noisy_s[j] = tmp * 1.
    return sentence


def threshold(x, t, positive_only=True):
    """returns, elementwise, 0 if between -t and t, x otherwise"""
    assert isinstance(t, float)
    if t > 0:
        if not positive_only:
            return F.threshold(x-t, 0, 0) - F.threshold(-x-t, 0, 0)
        else:
            return F.threshold(x-t, 0, 0)
    elif t < 0:
        raise ValueError("Threshold should be positive")
    else:
        return x


def tensorboard(niter_global, writer, gan_gen, gan_disc, autoencoder, log_freq):
    """Log gradients, weights, and some distributional features of the latent code"""
    if writer is None:
        return
    else:
        if niter_global % log_freq == 0:
            gan_gen.tensorboard(writer, niter_global) if gan_gen is not None else None
            gan_disc.tensorboard(writer, niter_global)
            autoencoder.tensorboard(writer, niter_global)


def activation_from_str(activation_str):
    """Returns a PyTorch activation from its lowercase str name"""
    assert activation_str in ['relu', 'lrelu'], 'Not implemented'
    activation = t.nn.ReLU() if activation_str == 'relu' else t.nn.LeakyReLU(.2)
    return activation


# modification of the PyTorch version to add a `maxlen` argument
def pad_packed_sequence(sequence, batch_first=False, maxlen=None):
    """Pads a packed batch of variable length sequences.

    It is an inverse operation to :func:`pack_padded_sequence`.

    The returned Variable's data will be of size TxBx*, where T is the length
    of the longest sequence and B is the batch size. If ``batch_first`` is True,
    the data will be transposed into BxTx* format.

    Batch elements will be ordered decreasingly by their length.

    Arguments:
        sequence (PackedSequence): batch to pad
        batch_first (bool, optional): if True, the output will be in BxTx*
            format.

    Returns:
        Tuple of Variable containing the padded sequence, and a list of lengths
        of each sequence in the batch.
    """
    var_data, batch_sizes = sequence
    max_batch_size = batch_sizes[0]
    output = var_data.data.new(len(batch_sizes) if maxlen is None else maxlen, max_batch_size, *var_data.size()[1:]).zero_()
    output = t.autograd.Variable(output)

    lengths = []
    data_offset = 0
    prev_batch_size = batch_sizes[0]
    for i, batch_size in enumerate(batch_sizes):
        output[i, :batch_size] = var_data[data_offset:data_offset + batch_size]
        data_offset += batch_size

        dec = prev_batch_size - batch_size
        if dec > 0:
            lengths.extend((i,) * dec)
        prev_batch_size = batch_size
    lengths.extend((i + 1,) * batch_size)
    lengths.reverse()

    if batch_first:
        output = output.transpose(0, 1)
    return output, lengths


def pack_padded_sequence(input, lengths, batch_first=False):
    """Packs a Variable containing padded sequences of variable length.

    Input can be of size ``TxBx*`` where T is the length of the longest sequence
    (equal to ``lengths[0]``), B is the batch size, and * is any number of
    dimensions (including 0). If ``batch_first`` is True ``BxTx*`` inputs are
    expected.

    The sequences should be sorted by length in a decreasing order, i.e.
    ``input[:,0]`` should be the longest sequence, and ``input[:,B-1]`` the
    shortest one.

    Note:
        This function accept any input that has at least two dimensions. You
        can apply it to pack the labels, and use the output of the RNN with
        them to compute the loss directly. A Variable can be retrieved from
        a :class:`PackedSequence` object by accessing its ``.data`` attribute.

    Arguments:
        input (Variable): padded batch of variable length sequences.
        lengths (list[int]): list of sequences lengths of each batch element.
        batch_first (bool, optional): if True, the input is expected in BxTx*
            format.

    Returns:
        a :class:`PackedSequence` object
    """
    print('start')
    if lengths[-1] <= 0:
        raise ValueError("length of all samples has to be greater than 0, "
                         "but found an element in 'lengths' that is <=0")
    if batch_first:
        input = input.transpose(0, 1)
    print('verif done')

    steps = []
    batch_sizes = []
    lengths_iter = reversed(lengths)
    current_length = next(lengths_iter)
    batch_size = input.size(1)
    if len(lengths) != batch_size:
        raise ValueError("lengths array has incorrect size")

    print('init done\nstart loop')
    for step, step_value in enumerate(input, 1):
        steps.append(step_value[:batch_size])
        batch_sizes.append(batch_size)

        while step == current_length:
            try:
                new_length = next(lengths_iter)
            except StopIteration:
                current_length = None
                break

            if current_length > new_length:  # remember that new_length is the preceding length in the array
                raise ValueError("lengths array has to be sorted in decreasing order")
            batch_size -= 1
            current_length = new_length
        if current_length is None:
            break
    return PackedSequence(t.cat(steps), batch_sizes)


def variable(array, requires_grad=False, to_float=True, cuda=True, volatile=False, gpu_id=None):
    """Wrapper for t.autograd.Variable"""
    if isinstance(array, np.ndarray):
        vv = t.from_numpy(array)
        vv = vv.cuda(gpu_id) if cuda else vv
        v = t.autograd.Variable(vv, requires_grad=requires_grad, volatile=volatile)
    elif isinstance(array, list) or isinstance(array, tuple):
        vv = t.from_numpy(np.array(array))
        vv = vv.cuda(gpu_id) if cuda else vv
        v = t.autograd.Variable(vv, requires_grad=requires_grad, volatile=volatile)
    elif isinstance(array, float) or isinstance(array, int):
        vv = t.from_numpy(np.array([array]))
        vv = vv.cuda(gpu_id) if cuda else vv
        v = t.autograd.Variable(vv, requires_grad=requires_grad, volatile=volatile)
    elif isinstance(array, t.Tensor) or isinstance(array, t.FloatTensor) or isinstance(array, t.DoubleTensor) or isinstance(array, t.LongTensor) or isinstance(array, t.cuda.FloatTensor) or isinstance(array, t.cuda.DoubleTensor) or isinstance(array,
                                                                                                                                                                                                                                                  t.cuda.LongTensor):
        v = t.autograd.Variable(array.cuda(gpu_id) if cuda else array, requires_grad=requires_grad, volatile=volatile)
    elif isinstance(array, t.autograd.Variable):
        v = array.cuda(gpu_id) if cuda else array
    else:
        raise ValueError("type(array): %s" % type(array))
    if to_float:
        return v.float()
    else:
        return v


def to_gpu(gpu, var, gpu_id=None):
    if gpu:
        return var.cuda(gpu_id)
    return var


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word2idx['<pad>'] = 0
        self.word2idx['<sos>'] = 1
        self.word2idx['<eos>'] = 2
        self.word2idx['<oov>'] = 3
        self.wordcounts = {}

    # to track word counts
    def add_word(self, word):
        if word not in self.wordcounts:
            self.wordcounts[word] = 1
        else:
            self.wordcounts[word] += 1

    # prune vocab based on count k cutoff or most frequently seen k words
    def prune_vocab(self, k=5, cnt=False):
        # get all words and their respective counts
        vocab_list = [(word, count) for word, count in self.wordcounts.items()]
        if cnt:
            # prune by count
            self.pruned_vocab = \
                {pair[0]: pair[1] for pair in vocab_list if pair[1] > k}
        else:
            # prune by most frequently seen words
            vocab_list.sort(key=lambda x: (x[1], x[0]), reverse=True)
            k = min(k, len(vocab_list))
            self.pruned_vocab = [pair[0] for pair in vocab_list[:k]]
        # sort to make vocabulary determistic
        self.pruned_vocab.sort()

        # add all chosen words to new vocabulary/dict
        for word in self.pruned_vocab:
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
        print("original vocab {}; pruned to {}".
              format(len(self.wordcounts), len(self.word2idx)))
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def __len__(self):
        return len(self.word2idx)


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


class SyntheticCorpus(object):
    def __init__(self, train, test, maxlen, vocab_size=11000, lowercase=False):
        """
        :param train, test: list of lists of integers
        """
        self.dictionary = Dictionary()
        self.maxlen = maxlen
        self.lowercase = lowercase
        self.vocab_size = vocab_size

        self.train = train
        self.test = test


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


def length_sort(items, lengths, descending=True):
    """In order to use pytorch variable length sequence package"""
    items = list(zip(items, lengths))
    items.sort(key=lambda x: x[1], reverse=True)
    items, lengths = zip(*items)
    return list(items), list(lengths)


def train_ngram_lm(kenlm_path, data_path, output_path, N):
    """
    Trains a modified Kneser-Ney n-gram KenLM from a text file.
    Creates a .arpa file to store n-grams.
    """
    # create .arpa file of n-grams
    curdir = os.path.abspath(os.path.curdir)
    #
    command = "bin/lmplz -o " + str(N) + " <" + os.path.join(curdir, data_path) + \
              " >" + os.path.join(curdir, output_path)
    command = command.replace("./ ", "") + " --discount_fallback"
    os.system("cd " + os.path.join(kenlm_path, 'build') + " && " + command)

    load_kenlm()
    # create language model
    model = kenlm.Model(output_path)

    return model


def get_ppl(lm, sentences):
    """
    Assume sentences is a list of strings (space delimited sentences)
    """
    total_nll = 0
    total_wc = 0
    for sent in sentences:
        words = sent.strip().split()
        score = lm.score(sent, bos=True, eos=False)
        word_count = len(words)
        total_wc += word_count
        total_nll += score
    ppl = 10 ** -(total_nll / total_wc)
    return ppl


def retokenize_data_for_vocab_size(data, unk_token=3, vocab_size=10000):
    # data in format of list of lists. outer list for each sentence. inner list contains
    # words as int.
    def retokenize_sentence(sentence, vocab_size):
        get_int_token = lambda w, v: w if (w <= v) else unk_token
        return [get_int_token(word, vocab_size) for word in sentence]

    data = [retokenize_sentence(sentence, vocab_size) for sentence in data]
    return data


def l2normalize(v, dim=None, eps=1e-12):
    if dim is None:
        return v / (v.norm() + eps)
    else:
        return v / (v.norm(dim=dim, keepdim=True).expand_as(v) + eps)


def save_nnlm(model, path):
    t.save(model.state_dict(), path)


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