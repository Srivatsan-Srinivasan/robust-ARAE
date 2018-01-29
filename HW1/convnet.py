import torch as t
from torch.nn import Conv1d as conv, MaxPool1d as maxpool, Linear as fc, Softmax, ReLU, Dropout, Tanh, BatchNorm1d as BN
from HW1.utils import variable
import torchtext
from torchtext.vocab import Vectors, GloVe


softmax = Softmax()
dropout = Dropout()
relu = ReLU()
tanh = Tanh()


def vectorize(text, TEXT, vdim=300):
    length, batch_size = text.data.numpy().shape
    return t.cat([TEXT.vocab.vectors[text.long().data.transpose(0,1)[i]].view(1,length,vdim) for i in range(batch_size)]).permute(0,2,1)


class Convnet(t.nn.Module):
    """Note that optimizer is hardcoded to Adam"""

    def __init__(self, learning_rate):
        super(Convnet, self).__init__()
        self.conv1 = conv(300, 100, 5, padding=2)
        #         self.conv2 = conv(100, 100, 3, padding=1)
        self.fc1 = fc(100, 100)
        self.fc2 = fc(100, 2)
        self.optimizer = t.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = t.nn.NLLLoss(size_average=True)

    def forward(self, x):
        dropout = Dropout(.2)
        xx = self.conv1(x)
        #         xx = tanh(xx)
        #         xx = self.conv2(xx)
        xx = tanh(t.max(xx, -1)[0])
        xx = self.fc1(xx)
        #         xx = dropout(xx)
        xx = tanh(xx)
        xx = self.fc2(xx)
        xx = dropout(xx)
        return softmax(xx)

    def eval_perf(self, iterator, TEXT, vdim=300):
        count = 0
        bs = iterator.batch_size * 1
        iterator.batch_size = 1
        for i, batch in enumerate(iterator):
            # get data
            text = batch.text
            y_pred = (self.forward(variable(vectorize(text, TEXT, vdim=vdim)))[:, 1] > 0.5).long()
            y = batch.label.long() * (-1) + 2

            count += t.sum((y == y_pred).long())
            if i >= len(iterator) - 1:
                break
        iterator.batch_size = bs
        return (count.float() / (bs*len(iterator))).data.numpy()[0]

    def fit(self, train_iter, val_iter, n_epochs, TEXT, vdim=300):

        for _ in range(n_epochs):
            for i, batch in enumerate(train_iter):
                # get data
                text = batch.text
                y_pred = self.forward(variable(vectorize(text, TEXT, vdim=vdim)))
                y = batch.label.long() * (-1) + 2

                # initialize gradients
                self.optimizer.zero_grad()

                # loss
                loss = self.loss.forward(y_pred, y)

                # compute gradients
                loss.backward()

                # update weights
                self.optimizer.step()

                if i >= len(train_iter) - 1:
                    break
            train_iter.init_epoch()
            self.eval()
            print("Validation accuracy after %d epochs: %.2f" % (_, self.eval_perf(val_iter, TEXT, vdim=vdim)))
            self.train()


def main(n_epochs, learning_rate, vdim=300):
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

    # loss and optimizer
    convnet = Convnet(learning_rate)
    convnet.fit(train_iter, val_iter, n_epochs, TEXT, vdim=vdim)

    upload = []
    true = []
    for batch in test_iter:
        # Your prediction data here (don't cheat!)
        probs = (convnet.forward(variable(vectorize(batch.text, TEXT, vdim=vdim)))[:, 1] > 0.5).long()
        upload += list(probs.data)
        true += batch.label.data.numpy().tolist()
    true = [x if x == 1 else 0 for x in true]
    print("test accuracy:")
    print(sum([(x==y) for x,y in zip(upload,true)])/ len(upload))


if __name__ == '__main__':
    learning_rate = 1e-4
    n_epochs = 25
    main(n_epochs, learning_rate)
