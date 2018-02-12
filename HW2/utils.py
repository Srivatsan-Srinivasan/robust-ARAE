import numpy as np
import torch as t
from torchtext.vocab import Vectors, GloVe
import pickle
import os
from sklearn.utils import shuffle
import torchtext

os.chdir('../HW2')  # so that there is not an import bug if the working directory isn't already HW2
from const import *


def variable(array, requires_grad=False, to_float=True, cuda=CUDA_DEFAULT):
    """Wrapper for t.autograd.Variable"""
    #import pdb; pdb.set_trace()
    if isinstance(array, np.ndarray):
        v = t.autograd.Variable(t.from_numpy(array), requires_grad=requires_grad)
    elif isinstance(array, list) or isinstance(array, tuple):
        v = t.autograd.Variable(t.from_numpy(np.array(array)), requires_grad=requires_grad)
    elif isinstance(array, float) or isinstance(array, int):
        v = t.autograd.Variable(t.from_numpy(np.array([array])), requires_grad=requires_grad)
    elif isinstance(array, t.Tensor) or isinstance(array, t.FloatTensor) or isinstance(array, t.DoubleTensor) or isinstance(array, t.LongTensor):
        v = t.autograd.Variable(requires_grad=requires_grad)            
    elif isinstance(array, t.autograd.Variable):
        v = array
    else:
        raise ValueError("type(array): %s" % type(array))
    if cuda:
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
