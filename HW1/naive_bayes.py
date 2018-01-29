from collections import Counter
import torch as t
import numpy as np
import torchtext
from torchtext.vocab import Vectors, GloVe


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


def main():
    # Text text processing library and methods for pretrained word embeddings

    # Our input $x$
    TEXT = torchtext.data.Field()

    # Our labels $y$
    LABEL = torchtext.data.Field(sequential=False)

    train_dataset, val_dataset, test_dataset = torchtext.datasets.SST.splits(
        TEXT, LABEL,
        filter_pred=lambda ex: ex.label != 'neutral')

    TEXT.build_vocab(train_dataset)
    LABEL.build_vocab(train_dataset)

    train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
        (train_dataset, val_dataset, test_dataset), batch_size=10, device=-1)

    url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
    TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))

    W, b = train_NB(TEXT, train_iter)

    upload = []
    true = []
    for batch in test_iter:
        # Your prediction data here (don't cheat!)
        probs = predict_NB(batch.text, W, b).long()
        upload += list(probs.data)
        true += batch.label.data.numpy().tolist()
    true = [x if x == 1 else -1 for x in true]
    print("test accuracy:")
    print(sum([(x*y == 1) for x,y in zip(upload,true)])/ len(upload))


if __name__ == '__main__':
    main()
