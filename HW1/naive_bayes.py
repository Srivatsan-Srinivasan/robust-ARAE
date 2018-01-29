from collections import Counter
import torch as t
import numpy as np


def predict_NB(text, W, bias):
    """
    sign(Wx + b)
    b is a torch variable
    W is a torch embedding
    """
    return t.sign(t.cat([t.sum(W(text.transpose(0, 1)[i])) for i in range(text.data.numpy().shape[1])]) + bias).long()


def train_NB(TEXT, train_iter, alpha=1):
    """
    get the coefficients of the naive bayes model
    Used only unigrams
    :param TEXT: a torchtext.data.Field instance
    :param train_iter: the training iterator. Obtained with torchtext.data.BucketIterator.splits
    :param alpha: the pseudo counts
    :return: W, b (word embedding, bias (torch variable))
    """
    positive_counts = Counter()
    negative_counts = Counter()
    W = t.nn.Embedding(len(TEXT.vocab), 1)
    i = 0
    pos = 0
    neg = 0

    # count the occurrences of each word in each class
    for b in train_iter:
        i += 1
        pos_tmp = t.nonzero((b.label == 1).data.long()).numpy().flatten().shape[0]
        neg_tmp = t.nonzero((b.label == 2).data.long()).numpy().flatten().shape[0]
        pos += pos_tmp
        neg += neg_tmp
        if neg_tmp < 10:
            positive_counts += Counter(b.text.transpose(0, 1).index_select(0, t.nonzero((b.label == 1).data.long()).squeeze()).data.numpy().flatten().tolist())
        if pos_tmp < 10:
            negative_counts += Counter(b.text.transpose(0, 1).index_select(0, t.nonzero((b.label == 2).data.long()).squeeze()).data.numpy().flatten().tolist())
        if i >= len(train_iter):
            break

    for k in range(len(TEXT.vocab)):  # pseudo counts
        positive_counts[k] += alpha
        negative_counts[k] += alpha

    # rescale
    scale_pos = sum(list(positive_counts.values()))
    scale_neg = sum(list(negative_counts.values()))
    positive_prop = {k: v / scale_pos for k, v in positive_counts.items()}
    negative_prop = {k: v / scale_neg for k, v in negative_counts.items()}

    r = {k: np.log(positive_prop[k] / negative_prop[k]) for k in range(len(TEXT.vocab))}
    W.weight.data = t.from_numpy(np.array([r[k] for k in range(len(TEXT.vocab))]))
    bias = np.log(pos / neg)

    return W, bias
