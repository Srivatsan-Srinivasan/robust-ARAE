import numpy as np
import torch as t
from torchtext.vocab import Vectors, GloVe
import pickle
import os

os.chdir('../HW2')  # so that there is not an import bug if the working directory isn't already HW2
from const import *


def variable(array, requires_grad=False, to_float=True, cuda=CUDA_DEFAULT):
    """Wrapper for t.autograd.Variable"""
    if isinstance(array, np.ndarray):
        v = t.autograd.Variable(t.from_numpy(array), requires_grad=requires_grad)
    elif isinstance(array, list) or isinstance(array, tuple):
        v = t.autograd.Variable(t.from_numpy(np.array(array)), requires_grad=requires_grad)
    elif isinstance(array, float) or isinstance(array, int):
        v = t.autograd.Variable(t.from_numpy(np.array([array])), requires_grad=requires_grad)
    elif isinstance(array, t.Tensor) or isinstance(array, t.FloatTensor) or isinstance(array, t.DoubleTensor) or isinstance(array, t.LongTensor):
        v = t.autograd.Variable(array, requires_grad=requires_grad)
    elif isinstance(array, t.autograd.Variable):
        return array
    else:
        raise ValueError("type(array): %s" % type(array))
    if cuda:
        v = v.cuda()
    if to_float:
        return v.float()
    else:
        return v


def embed_sentence(batch, TEXT, vec_dim=300, sentence_length=16):
    """Convert integer-encoded sentence to word vector representation"""
    return t.cat([TEXT.vocab.vectors[batch.text.data.long()[:, i]].view(1, sentence_length, vec_dim) for i in range(batch.batch_size)])


def batch2text(batch, TEXT):
    return " ".join([TEXT.vocab.itos[i] for i in batch.text[:, 0].data])


def load_embedding(TEXT):
    """
    By default it loads GloVe because it works well.
    It could be implemented so that it could load any other one but I think this is not very important
    @:param TEXT: torchtext.data.Field()
    """
    TEXT.vocab.load_vectors(vectors=GloVe())


def data_generator(train_iterator, model_str, context_size, cuda=True):
    """
    A generator that yields (x, target) couples. x is the input of the model, and target is the output (the next word)
    We need that because the last target of the current batch is the first word of the next batch
    Also, if not using a RNN, the prediction for the first few words necessitates to have the last words of the last batch, so that x is a bit
    different

    :param context_size: the size of the context size. None for RNN, an integer for NNLM
    :param model_str: the kind of model you want to use. See const.models and/or language_models.py for details
    :param cuda: whether to use GPU or not
    :param train_iterator: the torchtext training iterator
    :yield: (x, y)
    """
    for i, next_batch in enumerate(train_iterator):
        if i == 0:
            current_batch = next_batch
        else:
            if model_str == 'NNLM':
                if context_size is not None:
                    if i > 1:
                        starting_words = last_batch.text.transpose(0, 1)[:, -context_size:]
                    else:
                        starting_words = t.zeros(current_batch.text.size(1), context_size).float()
                    x = t.cat([variable(starting_words, to_float=False, cuda=cuda).long(), current_batch.text.transpose(0, 1).long()], 1)
                else:
                    raise ValueError('`context_size` should not be None')
            else:
                x = current_batch.text.transpose(0, 1).long()

            # you need the next batch first word to know what the target of the last word of the current batch is
            ending_word = next_batch.text.transpose(0, 1)[:, :1]
            target = t.cat([current_batch.text.transpose(0, 1)[:, 1:], ending_word], 1)

            last_batch = current_batch
            current_batch = next_batch

            yield x, target


def generate_inp_out(model_str, i, next_batch, last_batch, current_batch,
                     context_size=None, cuda=False):
    if i == 0:
        current_batch = next_batch
    else:
        if model_str == 'NNLM' and context_size is not None:
            if i > 1:
                starting_words = last_batch.text.transpose(0, 1)[:, -context_size:]
            else:
                starting_words = t.zeros(current_batch.text.size(1), context_size).float()
            x = t.cat([variable(starting_words, to_float=False, cuda=cuda).long(), current_batch.text.transpose(0, 1).long()], 1)
        else:
            x = current_batch.text.transpose(0, 1).long()

        # you need the next batch first word to know what the target of the last word of the current batch is
        ending_word = next_batch.text.transpose(0, 1)[:, :1]
        target = t.cat([current_batch.text.transpose(0, 1)[:, 1:], ending_word], 1)

        last_batch = current_batch
        current_batch = next_batch

        return x, target, last_batch, current_batch


def pickle_entry(entry, name):
    pickle.dump(entry, open(name + ".p", "wb"))


def load_pickle_entry(file_name):
    return pickle.load(open(file_name, "rb"))


if __name__ == '__main__':
    # test
    variable(np.zeros((10, 10)))
