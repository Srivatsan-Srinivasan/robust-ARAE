from collections import Counter
import numpy as np
from torch.nn import Sigmoid
from HW1.utils import variable
import torch as t
import torchtext
from torchtext.vocab import Vectors, GloVe


def predict_LR(batch, W, b):
    """

    :param batch: a Batch object. Comes from the torchtext iterators. It has a text attribute
    :param W: a torch embedding (the coefficients of the logistic regression)
    :param b: a torch variable (the bias)
    :return:
    """
    return Sigmoid()(t.cat([W(batch.text.transpose(0, 1)[i]).sum() for i in range(batch.text.data.numpy().shape[1])]) + b.float())


def eval_perf(iterator, W, b):
    """Evaluate the performance, typically on a validation iterator"""
    count = 0
    bs = iterator.batch_size * 1
    iterator.batch_size = 1
    for i, batch in enumerate(iterator):
        # get data
        y_pred = (Sigmoid()(t.cat([W(batch.text.transpose(0,1)[i]).sum() for i in range(batch.text.data.numpy().shape[1])]) + b.float()) > 0.5).long()
        y = batch.label.long()*(-1) + 2

        count += t.sum((y == y_pred).long())
        if i >= len(iterator) - 1:
            break
    iterator.batch_size = bs
    return (count.float() / (bs*len(iterator))).data.numpy()[0]


def train_LR(train_iter, val_iter, n_epochs, TEXT, learning_rate):
    """
    Default optim is RMSProp
    :param train_iter:
    :param val_iter:
    :param n_epochs:
    :param TEXT:
    :param learning_rate:
    :return: W, b (embedding, Variable)
    """
    W = t.nn.Embedding(len(TEXT.vocab), 1)
    b = variable(0., requires_grad=True, to_float=False)
    # loss and optimizer
    nll = t.nn.NLLLoss(size_average=True)

    optimizer = t.optim.RMSprop([b, W.weight], lr=learning_rate)
    sig = t.nn.Sigmoid()

    for _ in range(n_epochs):
        for i, batch in enumerate(train_iter):
            # get data
            y_pred = sig(t.cat([W(batch.text.transpose(0, 1)[i]).sum() for i in range(batch.text.data.numpy().shape[1])]) + b.float()).unsqueeze(1)
            y = batch.label.long() * (-1) + 2

            # initialize gradients
            optimizer.zero_grad()

            # loss
            y_pred = t.cat([1 - y_pred, y_pred], 1).float()  # nll needs two inputs: the prediction for the negative/positive classes

            loss = nll.forward(y_pred, y)

            # compute gradients
            loss.backward()

            # update weights
            optimizer.step()

            if i >= len(train_iter) - 1:
                break
        train_iter.init_epoch()
        print("Validation accuracy after %d epochs: %.2f" % (_, eval_perf(val_iter, W, b)))

    return W, b


def main(n_epochs, learning_rate):
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

    W, b = train_LR(train_iter, val_iter, n_epochs, TEXT, learning_rate)

    upload = []
    true = []
    for batch in test_iter:
        # Your prediction data here (don't cheat!)
        probs = (predict_LR(batch, W, b) > 0.5).long()
        upload += list(probs.data)
        true += batch.label.data.numpy().tolist()
    true = [x if x == 1 else 0 for x in true]
    print("test accuracy:")
    print(sum([(x == y) for x,y in zip(upload,true)])/ len(upload))


if __name__ == '__main__':
    learning_rate = 1e-2
    n_epochs = 25
    main(n_epochs, learning_rate)
