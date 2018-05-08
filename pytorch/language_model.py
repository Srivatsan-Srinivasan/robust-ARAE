import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from utils import *
import torch.nn.functional as F


class NNLM(t.nn.Module):
    def __init__(self, params):
        print(params)
        super(NNLM, self).__init__()
        print("Initializing LSTM")
        self.cuda_flag = params.get('cuda')
        self.model_str = 'LSTM'
        self.params = params

        # Initialize hyperparams.
        self.hidden_dim = params.get('hidden_dim')
        self.batch_size = params.get('batch_size')
        self.embedding_dim = params.get('embedding_dim')
        self.num_layers = params.get('num_layers')
        self.dropout = params.get('dropout')
        self.embed_dropout = params.get('embed_dropout')
        self.ntokens = params.get('ntokens')

        # Initialize embeddings.
        self.word_embeddings = t.nn.Embedding(self.ntokens, self.embedding_dim)
        self.hidden2out = nn.Linear(self.hidden_dim, self.ntokens)

        # Initialize network modules.
        self.model_rnn = nn.LSTM(self.embedding_dim, self.hidden_dim, dropout=self.dropout, num_layers=self.num_layers, batch_first=True)

        if self.embed_dropout:
            self.dropout_1 = nn.Dropout(self.dropout)
        self.dropout_2 = nn.Dropout(self.dropout)

    def init_hidden(self, batch_size=None):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim). The helper function
        # will return torch variable.
        return tuple((
            variable(np.zeros((self.num_layers, self.batch_size if batch_size is None else batch_size, self.hidden_dim)), cuda=self.cuda_flag, gpu_id=self.gpu_id),
            variable(np.zeros((self.num_layers, self.batch_size if batch_size is None else batch_size, self.hidden_dim)), cuda=self.cuda_flag, gpu_id=self.gpu_id)
        ))

    def forward(self, x_batch, lengths):

        # EMBEDDING
        embeds = self.word_embeddings(x_batch)
        # going from ` batch_size x bptt_length x embed_dim` to `bptt_length x batch_size x embed_dim`
        if self.embed_dropout:
            embeds = self.dropout_1(embeds)
        packed_embeddings = pack_padded_sequence(input=embeds,
                                                 lengths=lengths,
                                                 batch_first=True)

        # RECURRENT
        self.model_rnn.flatten_parameters()
        packed_output, state = self.model_rnn(packed_embeddings)
        rnn_out, _ = pad_packed_sequence(packed_output, batch_first=True, maxlen=max(lengths))
        rnn_out = self.dropout_2(rnn_out)

        # OUTPUT
        out_linear = self.hidden2out(rnn_out)
        return out_linear, state

    def get_ppl(self, indices, lengths):
        output, _ = self.forward(indices, lengths)

        # mask the pad tokens
        mask = indices.gt(0)  # gt: greater than.
        masked_indices = indices.masked_select(mask)  # it flattens the output to n_examples x sentence_length
        output_mask = mask.unsqueeze(2).expand(mask.size(0), mask.size(1), self.ntokens)  # replicate the mask for each vocabulary word. Size batch_size x |V|
        flattened_output = output.view(-1, self.ntokens)
        output_mask = output_mask.contiguous().view(-1, self.ntokens)
        masked_output = flattened_output.masked_select(output_mask).view(-1, self.ntokens)  # batch_size x max_len classification problems, without the padding
        loss = F.cross_entropy(masked_output, masked_indices)  # batch_size x max_len classification problems

        ppl = t.exp(loss)
        return ppl.cpu().data.numpy()[0]

    def generate(self, batch_size, maxlen, gpu_id=None, gpu=True, sample=True, temp=1.):
        state = self.init_hidden(batch_size)

        # <sos>
        start_symbols = variable(t.ones(10, 1).long(), to_float=False, cuda=gpu, volatile=False, gpu_id=gpu_id)
        start_symbols.data.resize_(batch_size, 1)
        start_symbols.data.fill_(1)

        embedding = self.word_embeddings(start_symbols)
        inputs = embedding * 1.

        # unroll
        all_indices = []
        for i in range(maxlen):
            output, state = self.model_rnn(inputs, state)
            output = output.float()
            state = state[0].detach(), state[1].detach()
            overvocab = self.hidden2out(output.squeeze(1))

            if not sample:
                vals, indices = t.max(overvocab, 1)  # this is not an error on newer PyTorch
            else:
                # sampling
                probs = F.softmax(overvocab / temp, dim=1)
                indices = t.multinomial(probs, 1)

            all_indices.append(indices)
            embedding = self.word_embeddings(indices)
            inputs = embedding * 1.

        max_indices = t.stack(all_indices, 1)
        return max_indices.squeeze()

    def __cuda__(self, gpu_id):
        self.cuda(gpu_id)
        self.gpu_id = gpu_id


def generate_synthetic_dataset(oracle, n, maxlen, gpu, gpu_id=None, add_eos=True, add_sos=True, n_per_batch=500):
    # Init
    dataset = []

    while len(dataset) <= n:

        # Generate sentences, one batch at a time
        x = oracle.generate(n_per_batch, maxlen, gpu=gpu, gpu_id=gpu_id, sample=True, temp=1.).long()

        # Add EOS and SOS tokens
        if add_sos:
            x = t.cat([variable(t.ones(n_per_batch, 1), to_float=False, cuda=gpu, gpu_id=gpu_id).long(), x], 1)
        if add_eos:
            x = t.cat([x, variable(2 * t.ones(n_per_batch, 1), to_float=False, cuda=gpu, gpu_id=gpu_id).long()], 1)

        dataset += x.data.cpu().numpy().tolist()

    return dataset, oracle


class TemporalCrossEntropyLoss(t.nn.modules.loss._WeightedLoss):
    r"""This criterion combines `LogSoftMax` and `NLLLoss` in one single class.

    It is useful when training a temporal classification problem with `C` classes over time series of length `T`.
    If provided, the optional argument `weight` should be a 1D `Tensor`
    assigning weight to each of the classes.
    This is particularly useful when you have an unbalanced training set.

    The `input` is expected to contain scores for each class.

    `input` has to be a 3D `Tensor` of size `(minibatch, C, T)`.

    This criterion expects a class index (0 to C-1) as the
    `target` for each value of a 2D tensor of size `(minibatch, T)`

    The loss can be described as, for each time step `t`::

        loss(x, class, t) = -log(exp(x[class]) / (\sum_j exp(x[j])))
                       = -x[class] + log(\sum_j exp(x[j]))
    The total loss being:
        loss(x, class) = \sum_t loss(x, class, t)


    or in the case of the `weight` argument being specified::

        loss(x, class, t) = weight[class] * (-x[class] + log(\sum_j exp(x[j])))

    The losses are averaged across observations for each minibatch.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each class.
           If given, has to be a Tensor of size "C"
        size_average (bool, optional): By default, the losses are averaged over observations for each minibatch.
           However, if the field size_average is set to ``False``, the losses are
           instead summed for each minibatch. Ignored if reduce is ``False``.
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When size_average is
            ``True``, the loss is averaged over non-ignored targets.

    Shape:
        - Input: :math:`(N, C, T)` where `C = number of classes` and `T = sentence length`
        - Target: :math:`(N, T)` where each value is `0 <= targets[i] <= C-1`
        - Output: scalar. If reduce is ``False``, then :math:`(N)` instead.

    Examples::

        >>> loss = nn.CrossEntropyLoss2D()
        >>> input = variable(torch.randn(batch_size, vocab_size, sentence_length), requires_grad=True)  # for each element of the batch, for each position x=0..sentence_length-1, it gives a probability distribution over the |V| possible words
        >>> target = variable(...)  # size (batch_size, sentence_length). LongTensor containing the correct class (correct next word) at each position of the sentence
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self, weight=None, size_average=True, ignore_index=-100):
        """Average over the batch_size"""
        super(TemporalCrossEntropyLoss, self).__init__(weight, size_average)
        self.ignore_index = ignore_index
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, size_average=size_average, ignore_index=ignore_index)

    def forward(self, pred, true):
        """
        Let `C` be the number of classes and `|V|` the vocab size
        What this class does is just reshaping the inputs so that you can use classical cross entropy on top of that

        :param pred: FloatTensor of shape (batch_size, |V|, C)
        :param true: LongTensor of shape (batch_size, C)
        :return:
        """
        t.nn.modules.loss._assert_no_grad(true)

        # doing it this way allows to use parallelism. Better than looping on last dim !
        # note that this version of pytorch seems outdated
        true_ = true.contiguous().view(true.size(0) * true.size(1))  # true.size() = (batch_size, bptt_length)
        pred_ = pred.contiguous().view(pred.size(0) * pred.size(2), pred.size(1))  # pred.size() = (batch_size, vocab_size, bptt_length)
        return self.cross_entropy.forward(pred_, true_)
