"""
For the synthetic experiment, you generate a dataset using a random LSTM (that has no EOS token)
You generate sentences of chosen lengths, and then you append the EOS token at the end.
Finally, you train your ARAE variant on this, and you can evaluate through training what the TRUE perplexity actually is
"""

import torch as t
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from utils import variable, to_gpu, pad_packed_sequence


class Oracle(t.nn.Module):
    def __init__(self, emsize, nhidden, ntokens, nlayers, gpu=False, gpu_id=None):
        """
        :param emsize: !!! DON'T INCLUDE THE EOS TOKEN
        """
        super(Oracle, self).__init__()
        self.emsize = emsize
        self.nhidden = nhidden
        self.ntokens = ntokens
        self.nlayers = nlayers
        self.gpu_id = gpu_id

        self.lstm = t.nn.LSTM(input_size=emsize,
                              hidden_size=nhidden,
                              num_layers=nlayers,
                              batch_first=True)
        self.gpu = gpu
        self.embedding = t.nn.Embedding(ntokens, emsize)
        self.linear = t.nn.Linear(nhidden, ntokens)
        self.start_symbols = to_gpu(gpu, variable(t.ones(10, 1).long(), to_float=False, cuda=False, volatile=False), gpu_id=gpu_id)

    def init_hidden(self, bsz):
        zeros1 = variable(t.zeros(self.nlayers, bsz, self.nhidden), gpu_id=self.gpu_id, cuda=self.gpu)
        zeros2 = variable(t.zeros(self.nlayers, bsz, self.nhidden), gpu_id=self.gpu_id, cuda=self.gpu)
        return (to_gpu(self.gpu, zeros1, gpu_id=self.gpu_id), to_gpu(self.gpu, zeros2, gpu_id=self.gpu_id))

    def forward(self, indices, lengths):
        embeddings = self.embedding(indices)
        packed_embeddings = pack_padded_sequence(input=embeddings,
                                                 lengths=lengths,
                                                 batch_first=True)
        hidden = self.init_hidden(indices.size(0))
        # Encode
        packed_output, state = self.lstm(packed_embeddings, hidden)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, maxlen=None)
        return self.linear(output.contiguous().view(-1, self.nhidden)).view(indices.size(0), output.size(1), self.ntokens)

    def generate(self, batch_size, maxlen, sample=True, temp=1.):
        state = self.init_hidden(batch_size)

        # <sos>
        self.start_symbols.data.resize_(batch_size, 1)
        self.start_symbols.data.fill_(1)

        embedding = self.embedding(self.start_symbols)
        inputs = embedding*1.

        # unroll
        all_indices = []
        for i in range(maxlen):
            output, state = self.lstm(inputs, state)
            overvocab = self.linear(output.squeeze(1))

            if not sample:
                vals, indices = t.max(overvocab, 1)  # this is not an error on newer PyTorch
            else:
                # sampling
                probs = F.softmax(overvocab / temp, dim=1)
                indices = t.multinomial(probs, 1)

            all_indices.append(indices)
            embedding = self.embedding(indices)
            inputs = embedding*1.

        max_indices = t.stack(all_indices, 1)
        return max_indices.squeeze() + 3  # avoid the 3 first tokens that are used for padding/start/end. Note that there is no OOV

    def get_ppl(self, indices, lengths):
        output = self.forward(indices, lengths)
        ppl = t.exp(F.cross_entropy(output, indices))
        return ppl


def generate_synthetic_dataset(lengths, emsize, nhidden, ntokens, nlayers, gpu, gpu_id=None,
                               add_eos=True, add_sos=True, oracle=None, n_per_batch=500, oov_proportion=0):
    """
    :param lengths: a dict {sentence_length: #sentences_of_that_length}
    :return: a synthetic dataset, with the format {length: tensor_containing_all_sentences_of_this_length}
    """
    # Init
    if oracle is None:
        oracle = Oracle(emsize, nhidden, ntokens, nlayers, gpu, gpu_id=gpu_id)
    dataset = {}

    for length, how_many in lengths.items():

        # Init
        dataset[length] = variable(t.zeros(how_many, length), to_float=False, gpu_id=oracle.gpu_id, cuda=gpu)

        # Generate sentences, one batch at a time
        for i in range(0, how_many, n_per_batch):
            if i + n_per_batch >= how_many:
                dataset[length][i:, :] = oracle.generate(how_many - i, length)
            else:
                generated = oracle.generate(n_per_batch, length)
                dataset[length][i:i + n_per_batch, :] = generated

        # Add OOV words
        dataset[length] = introduce_oov(dataset[length], oov_proportion, gpu, gpu_id=gpu_id)

        # Add EOS and SOS tokens
        if add_sos:
            dataset[length] = t.cat([variable(t.ones(how_many, 1), to_float=False, cuda=gpu, gpu_id=oracle.gpu_id).long(), dataset[length]], 1)
        if add_eos:
            dataset[length] = t.cat([dataset[length], variable(2*t.ones(how_many, 1), to_float=False, cuda=gpu, gpu_id=oracle.gpu_id).long()], 1)

    return dataset, oracle


def introduce_oov(dataset, proportion, gpu, gpu_id=None):
    """
    :param dataset: a variable representing N sentences of length L
    :param proportion: the proportion of OOV to add
    :return: the modified dataset
    """
    assert 0 <= proportion <= 1
    if proportion == 0:
        return dataset
    else:
        bernoullis = variable(t.bernoulli(proportion*t.ones(*dataset.size())).long(), to_float=False, cuda=gpu, gpu_id=gpu_id)
        new_dataset = dataset.long() + (3 - dataset.long())*bernoullis
        return new_dataset
