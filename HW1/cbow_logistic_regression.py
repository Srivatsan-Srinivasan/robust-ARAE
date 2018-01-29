import torch as t
from HW1.utils import variable
import numpy as np
from torch.nn import Sigmoid


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
    return (count.float() / len(iterator)).data.numpy()[0]


def train_CBOW(train_iter, val_iter, TEXT, learning_rate, n_epochs, vdim=300):
    W = variable(np.random.normal(0, .1, (vdim,)), True)
    b = variable(0., True)

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
        print("Validation accuracy after %d epochs: %.2f" % (_, eval_perf(val_iter)))

    return W, b


def predict_CBOW(batch, TEXT, W, b, vdim=300):
    text_ = batch.text
    return Sigmoid()(t.mm(variable(vectorize(text_, TEXT, vdim=vdim)), W.float().resize(vdim, 1)).squeeze() + b.float().squeeze())
