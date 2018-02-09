import numpy as np
import torch as t
from torchtext.vocab import Vectors, GloVe


def variable(array, requires_grad=False, to_float=True, cuda=False):
    """Wrapper for t.autograd.Variable"""
    if isinstance(array, np.ndarray):
        v = t.autograd.Variable(t.from_numpy(array), requires_grad=requires_grad)
    elif isinstance(array, list) or isinstance(array,tuple):
        v = t.autograd.Variable(t.from_numpy(np.array(array)), requires_grad=requires_grad)
    elif isinstance(array, float) or isinstance(array, int):
        v = t.autograd.Variable(t.from_numpy(np.array([array])), requires_grad=requires_grad)
    elif isinstance(array, t.Tensor) or isinstance(array, t.FloatTensor) or isinstance(array, t.DoubleTensor) or isinstance(array, t.LongTensor):
        v = t.autograd.Variable(array, requires_grad=requires_grad)
    else:
        raise ValueError
    if cuda:
        v = v.cuda()
    if to_float:
        return v.float()
    else:
        return v


def embed_sentence(batch, TEXT, vec_dim=300, sentence_length=16):
    """Convert integer-encoded sentence to word vector representation"""
    return t.cat([TEXT.vocab.vectors[batch.text.data.long()[:,i]].view(1,sentence_length,vec_dim) for i in range(batch.batch_size)])


def batch2text(batch, TEXT):
    return " ".join([TEXT.vocab.itos[i] for i in batch.text[:, 0].data])


def load_embedding(TEXT):
    """
    By default it loads GloVe because it works well.
    It could be implemented so that it could load any other one but I think this is not very important
    @:param TEXT: torchtext.data.Field()
    """
    TEXT.vocab.load_vectors(vectors=GloVe())
