import os
import numpy as np
import random
import torch as t


def load_kenlm():
    global kenlm
    import kenlm


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
        tmp = noisy_s[i]*1.
        noisy_s[i] = noisy_s[j]*1.
        noisy_s[j] = tmp*1.
    return sentence


def tensorboard(niter_global, writer, gan_gen, gan_disc, autoencoder, log_freq):
    """Log gradients, weights, and some distributional features of the latent code"""
    if writer is None:
        return
    else:
        if niter_global % log_freq == 0:
            gan_gen.tensorboard(writer, niter_global)
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


def variable(array, requires_grad=False, to_float=True, cuda=True, volatile=False):
    """Wrapper for t.autograd.Variable"""
    if isinstance(array, np.ndarray):
        vv = t.from_numpy(array)
        vv = vv.cuda() if cuda else vv
        v = t.autograd.Variable(vv, requires_grad=requires_grad, volatile=volatile)
    elif isinstance(array, list) or isinstance(array, tuple):
        vv = t.from_numpy(np.array(array))
        vv = vv.cuda() if cuda else vv
        v = t.autograd.Variable(vv, requires_grad=requires_grad, volatile=volatile)
    elif isinstance(array, float) or isinstance(array, int):
        vv = t.from_numpy(np.array([array]))
        vv = vv.cuda() if cuda else vv
        v = t.autograd.Variable(vv, requires_grad=requires_grad, volatile=volatile)
    elif isinstance(array, t.Tensor) or isinstance(array, t.FloatTensor) or isinstance(array, t.DoubleTensor) or isinstance(array, t.LongTensor) or isinstance(array, t.cuda.FloatTensor) or isinstance(array, t.cuda.DoubleTensor) or isinstance(array, t.cuda.LongTensor):
        v = t.autograd.Variable(array.cuda() if cuda else array, requires_grad=requires_grad, volatile=volatile)
    elif isinstance(array, t.autograd.Variable):
        v = array.cuda() if cuda else array
    else:
        raise ValueError("type(array): %s" % type(array))
    if to_float:
        return v.float()
    else:
        return v


def to_gpu(gpu, var):
    if gpu:
        return var.cuda()
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
    def __init__(self, path, maxlen, vocab_size=11000, lowercase=False):
        self.dictionary = Dictionary()
        self.maxlen = maxlen
        self.lowercase = lowercase
        self.vocab_size = vocab_size
        self.train_path = os.path.join(path, 'train.txt')
        self.test_path = os.path.join(path, 'test.txt')

        # make the vocabulary from training set
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


def batchify(data, bsz, shuffle=False, gpu=False):
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
        batch = data[i*bsz:(i+1)*bsz]
        # subtract 1 from lengths b/c includes BOTH starts & end symbols
        lengths = [len(x)-1 for x in batch]
        # sort items by length (decreasing)
        batch, lengths = length_sort(batch, lengths)

        # source has no end symbol
        source = [x[:-1] for x in batch]
        # target has no start symbol
        target = [x[1:] for x in batch]

        # find length to pad to
        maxlen = max(lengths)
        for x, y in zip(source, target):
            zeros = (maxlen-len(x))*[0]
            x += zeros
            y += zeros

        source = t.LongTensor(np.array(source))
        target = t.LongTensor(np.array(target)).view(-1)

        if gpu:
            source = source.cuda()
            target = target.cuda()

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
    command = "bin/lmplz -o "+str(N)+" <"+os.path.join(curdir, data_path) + \
              " >"+os.path.join(curdir, output_path)
    os.system("cd "+os.path.join(kenlm_path, 'build')+" && "+command)

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
    ppl = 10**-(total_nll/total_wc)
    return ppl
