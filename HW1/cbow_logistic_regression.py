import torch as t
from HW1.utils import variable
import numpy as np
from torch.nn import Sigmoid
import torchtext
from torchtext.vocab import Vectors, GloVe


def vectorize(text, TEXT, vdim=300):
    length, batch_size = text.data.numpy().shape
    return t.mean(t.cat([TEXT.vocab.vectors[text.long().data.transpose(0,1)[i]].view(1,length,vdim) for i in range(batch_size)]), 1)


def eval_perf(iterator, TEXT, vdim, W, b):
    count = 0
    bs = iterator.batch_size * 1
    iterator.batch_size = 1
    for i, batch in enumerate(iterator):
        # get data
        text_ = batch.text
        y_pred = (Sigmoid()(t.mm(variable(vectorize(text_, TEXT, vdim=vdim)),W.float().resize(300,1)).squeeze() + b.float().squeeze()) > 0.5).long()
        y = batch.label.long()*(-1) + 2

        count += t.sum((y == y_pred).long())
        if i >= len(iterator) - 1:
            break
    iterator.batch_size = bs
    return (count.float() / (bs*len(iterator))).data.numpy()[0]


def train_CBOW(train_iter, val_iter, TEXT, learning_rate, n_epochs, vdim=300):
    W = variable(np.random.normal(0, .1, (vdim,)), requires_grad=True, to_float=False)
    b = variable(0., requires_grad=True, to_float=False)  # if to_float is True, it raises an exception regarding non-leaf variables...

    # loss and optimizer
    nll = t.nn.NLLLoss(size_average=True)
    optimizer = t.optim.RMSprop([b, W], lr=learning_rate)
    sig = t.nn.Sigmoid()

    for _ in range(n_epochs):
        for i, batch in enumerate(train_iter):
            # get data
            text_ = batch.text
            length = text_.data.numpy().shape[0]
            y_pred = sig(t.mm(variable(vectorize(text_, TEXT, vdim=vdim)), W.float().resize(300, 1)).squeeze() + b.float().squeeze()).unsqueeze(1)
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
        print("Validation accuracy after %d epochs: %.2f" % (_, eval_perf(val_iter, TEXT, vdim, W, b)))

    return W, b


def predict_CBOW(batch, TEXT, W, b, vdim=300):
    text_ = batch.text
    return Sigmoid()(t.mm(variable(vectorize(text_, TEXT, vdim=vdim)), W.float().resize(vdim, 1)).squeeze() + b.float().squeeze())


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

    W, b = train_CBOW(train_iter, val_iter, TEXT, learning_rate, n_epochs)

    upload = []
    true = []
    for batch in test_iter:
        # Your prediction data here (don't cheat!)
        probs = (predict_CBOW(batch, TEXT, W, b) > 0.5).long()
        upload += list(probs.data)
        true += batch.label.data.numpy().tolist()
    true = [x if x == 1 else 0 for x in true]
    print("test accuracy:")
    print(sum([(x==y) for x,y in zip(upload,true)])/ len(upload))


if __name__ == '__main__':
    learning_rate = 1e-2
    n_epochs = 25
    main(n_epochs, learning_rate)
