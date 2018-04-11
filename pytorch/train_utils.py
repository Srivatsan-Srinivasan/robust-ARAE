import math
import time
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from models import Seq2Seq
from utils import to_gpu, variable, train_ngram_lm, get_ppl


def save_model(autoencoder, gan_gen, gan_disc, args):
    print("Saving models")
    with open('./output/{}/autoencoder_model.pt'.format(args.outf), 'wb') as f:
        torch.save(autoencoder.state_dict(), f)
    with open('./output/{}/gan_gen_model.pt'.format(args.outf), 'wb') as f:
        torch.save(gan_gen.state_dict(), f)
    with open('./output/{}/gan_disc_model.pt'.format(args.outf), 'wb') as f:
        torch.save(gan_disc.state_dict(), f)


def evaluate_autoencoder(autoencoder, corpus, criterion_ce, data_source, epoch, args):
    # Turn on evaluation mode which disables dropout.
    autoencoder.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary.word2idx)
    all_accuracies = 0
    bcnt = 0
    for i, batch in enumerate(data_source):
        source, target, lengths = batch
        source = to_gpu(args.cuda, Variable(source, volatile=True))
        target = to_gpu(args.cuda, Variable(target, volatile=True))

        mask = target.gt(0)
        masked_target = target.masked_select(mask)
        # examples x ntokens
        output_mask = mask.unsqueeze(1).expand(mask.size(0), ntokens)

        # output: batch x seq_len x ntokens
        output = autoencoder(source, variable(lengths, cuda=args.cuda, to_float=False).long(), noise=True)  # output = autoencoder(source, lengths, noise=True)
        flattened_output = output.view(-1, ntokens)

        masked_output = flattened_output.masked_select(output_mask).view(-1, ntokens)
        total_loss += criterion_ce(masked_output/args.temp, masked_target).data

        # accuracy
        max_vals, max_indices = torch.max(masked_output, 1)
        all_accuracies += torch.mean(max_indices.eq(masked_target).float()).data[0]
        bcnt += 1

        aeoutf = "./output/%s/%d_autoencoder.txt" % (args.outf, epoch)
        with open(aeoutf, "a") as f:
            max_values, max_indices = torch.max(output, 2)
            max_indices = max_indices.view(output.size(0), -1).data.cpu().numpy()
            target = target.view(output.size(0), -1).data.cpu().numpy()
            for t, idx in zip(target, max_indices):
                # real sentence
                chars = " ".join([corpus.dictionary.idx2word[x] for x in t])
                f.write(chars)
                f.write("\n")
                # autoencoder output sentence
                chars = " ".join([corpus.dictionary.idx2word[x] for x in idx])
                f.write(chars)
                f.write("\n\n")

    return total_loss[0] / len(data_source), all_accuracies/bcnt


def evaluate_generator(gan_gen, autoencoder, corpus, noise, epoch, args):
    gan_gen.eval()
    autoencoder.eval()

    # generate from fixed random noise
    fake_hidden = gan_gen(noise)
    autoencoder_ = autoencoder if args.n_gpus == 1 else autoencoder.module
    max_indices = autoencoder_.generate(fake_hidden, args.maxlen, sample=args.sample)

    with open("./output/%s/%s_generated.txt" % (args.outf, epoch), "w") as f:
        max_indices = max_indices.data.cpu().numpy()
        for idx in max_indices:
            # generated sentence
            words = [corpus.dictionary.idx2word[x] for x in idx]
            # truncate sentences to first occurrence of <eos>
            truncated_sent = []
            for w in words:
                if w != '<eos>':
                    truncated_sent.append(w)
                else:
                    break
            chars = " ".join(truncated_sent)
            f.write(chars)
            f.write("\n")


def train_lm(gan_gen, autoencoder, corpus, eval_path, save_path, args):
    """Evaluate the performance of a simple language model that is trained on synthetic sentences"""
    # generate 100000 examples
    indices = []
    noise = to_gpu(args.cuda, Variable(torch.ones(100, args.z_size)))
    for i in range(1000):
        noise.data.normal_(0, 1)

        fake_hidden = gan_gen(noise)
        max_indices = autoencoder.generate(fake_hidden, args.maxlen)
        indices.append(max_indices.data.cpu().numpy())

    indices = np.concatenate(indices, axis=0)

    # write generated sentences to text file
    with open(save_path+".txt", "w") as f:
        # laplacian smoothing
        for word in corpus.dictionary.word2idx.keys():
            f.write(word+"\n")
        for idx in indices:
            # generated sentence
            #print(idx[0],"###")
            words = [corpus.dictionary.idx2word[x[0]] for x in idx]

            # truncate sentences to first occurrence of <eos>
            truncated_sent = []
            for w in words:
                if w != '<eos>':
                    truncated_sent.append(w)
                else:
                    break
            chars = " ".join(truncated_sent)
            f.write(chars+"\n")

    # train language model on generated examples
    lm = train_ngram_lm(kenlm_path=args.kenlm_path,
                        data_path=save_path+".txt",
                        output_path=save_path+".arpa",
                        N=args.N)

    # load sentences to evaluate on
    with open(eval_path, 'r') as f:
        lines = f.readlines()
    sentences = [l.replace('\n', '') for l in lines]
    ppl = get_ppl(lm, sentences)

    return ppl


def train_ae(autoencoder, criterion_ce, optimizer_ae, train_data, batch, total_loss_ae, start_time, i, ntokens, epoch, args):
    """
    Train autoencoder
    :param batch: a 3-tuple (source_sentences, target_sentences, sentences_lengths)
                  Note that the source and the target are the same, but with an SOS for source and EOS for target
    """
    autoencoder.train()
    autoencoder.zero_grad()

    source, target, lengths = batch  # note that target is flattened
    source = to_gpu(args.cuda, Variable(source))  # source has no end symbol
    target = to_gpu(args.cuda, Variable(target))  # target has no start symbol

    # Create sentence length mask over padding
    mask = target.gt(0)  # gt: greater than. 0 is the padding idx. All other idx are greater than 0
    masked_target = target.masked_select(mask)  # it flattens the output to n_examples x sentence_length
    output_mask = mask.unsqueeze(1).expand(mask.size(0), ntokens)  # replicate the mask for each vocabulary word. Size batch_size x |V|

    # output: (batch_size, max_len, ntokens)
    output = autoencoder(source, variable(lengths, cuda=args.cuda, to_float=False).long(), noise=True)

    # output_size: (batch_size x max_len, ntokens)
    flattened_output = output.view(-1, ntokens)
    masked_output = flattened_output.masked_select(output_mask).view(-1, ntokens)  # batch_size x max_len classification problems, without the padding

    loss = criterion_ce(masked_output/args.temp, masked_target)  # batch_size x max_len classification problems
    loss.backward()

    # `clip_grad_norm` to prevent exploding gradient in RNNs / LSTMs
    torch.nn.utils.clip_grad_norm(autoencoder.parameters(), args.clip)
    optimizer_ae.step()

    total_loss_ae += loss.data

    accuracy = None
    if i % args.log_interval == 0 and i > 0:
        # accuracy
        probs = F.softmax(masked_output, 1)
        max_vals, max_indices = torch.max(probs, 1)
        accuracy = torch.mean(max_indices.eq(masked_target).float()).data[0]

        cur_loss = total_loss_ae[0] / args.log_interval
        elapsed = time.time() - start_time
        print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
              'loss {:5.2f} | ppl {:8.2f} | acc {:8.2f}'
              .format(epoch, i, len(train_data),
                      elapsed * 1000 / args.log_interval,
                      cur_loss, math.exp(cur_loss), accuracy))

        with open("./output/{}/logs.txt".format(args.outf), 'a') as f:
            f.write('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | acc {:8.2f}\n'.
                    format(epoch, i, len(train_data),
                           elapsed * 1000 / args.log_interval,
                           cur_loss, math.exp(cur_loss), accuracy))

        total_loss_ae = 0
        start_time = time.time()

    return total_loss_ae, start_time


def train_gan_g(gan_gen, gan_disc, optimizer_gan_g, args):
    """
    Note that the sign of the loss (choosing .backward(one) over .backward(mone)) doesn't matter, as long as there is
    consistency between G and D, AND between the two parts of the loss of D

    It is because f is 1-Lipschitz iff -f also is

    See the comment of martinarjovsky on:
        * https://github.com/martinarjovsky/WassersteinGAN/issues/9
        * https://cloud.githubusercontent.com/assets/5272722/22793339/9210a6ea-eebd-11e6-8f3d-aeae2827b955.png
    """
    one = to_gpu(args.cuda, torch.FloatTensor(args.n_gpus * [1]))
    mone = one * -1

    gan_gen.train()
    gan_gen.zero_grad()

    noise = to_gpu(args.cuda,
                   Variable(torch.ones(args.batch_size, args.z_size)))
    noise.data.normal_(0, 1)

    fake_hidden = gan_gen(noise)
    errG = gan_disc(fake_hidden)

    # loss / backprop
    errG.backward(one)
    optimizer_gan_g.step()

    return errG


def train_gan_d(autoencoder, gan_disc, gan_gen, optimizer_gan_d, optimizer_ae, batch, args, writer = None):
    """
    Note that the sign of the loss (choosing .backward(one) over .backward(mone)) doesn't matter, as long as there is
    consistency between G and D, AND between the two parts of the loss of D

    It is because f is 1-Lipschitz iff -f also is

    See the comment of martinarjovsky on:
        * https://github.com/martinarjovsky/WassersteinGAN/issues/9
        * https://cloud.githubusercontent.com/assets/5272722/22793339/9210a6ea-eebd-11e6-8f3d-aeae2827b955.png
    """
    one = to_gpu(args.cuda, torch.FloatTensor(args.n_gpus * [1]))
    mone = one * -1
    # clamp parameters to a cube
    if not args.gradient_penalty and not args.spectralnorm:
        for p in gan_disc.parameters():
            p.data.clamp_(-args.gan_clamp, args.gan_clamp)

    autoencoder.train()
    autoencoder.zero_grad()
    gan_disc.train()
    gan_disc.zero_grad()

    # positive samples ----------------------------
    # generate real codes
    source, target, lengths = batch
    source = to_gpu(args.cuda, Variable(source))
    target = to_gpu(args.cuda, Variable(target))

    # batch_size x nhidden
    real_hidden = autoencoder(source, variable(lengths, cuda=args.cuda, to_float=False).long(), noise=False, encode_only=True)
    grad_norm = sum(list(Seq2Seq.grad_norm.values()))
    real_hidden.register_hook(lambda grad: grad_hook(grad, grad_norm, args))

    # loss / backprop
    errD_real = gan_disc(real_hidden,writer=writer)
    errD_real.backward(one)

    # negative samples ----------------------------
    # generate fake codes
    noise = to_gpu(args.cuda,
                   Variable(torch.ones(args.batch_size, args.z_size)))
    noise.data.normal_(0, 1)

    # loss / backprop
    fake_hidden = gan_gen(noise)
    errD_fake = gan_disc(fake_hidden.detach())
    errD_fake.backward(mone)
    if args.gradient_penalty:
        errD_grad = gan_disc.gradient_penalty(real_hidden, fake_hidden)
        errD_grad.backward(one)

    # `clip_grad_norm` to prevent exploding gradient problem in RNNs / LSTMs
    torch.nn.utils.clip_grad_norm(autoencoder.parameters(), args.clip)

    optimizer_gan_d.step()
    optimizer_ae.step()
    errD = -(errD_real - errD_fake)

    return errD, errD_real, errD_fake


def grad_hook(grad, grad_norm, args):
    # Gradient norm: regularize to be same
    # code_grad_gan * code_grad_ae / norm(code_grad_gan)
    if args.enc_grad_norm:
        gan_norm = torch.norm(grad, 2, 1).detach().data.mean()
        # grad_norm = autoencoder.grad_norm
        normed_grad = grad * grad_norm / gan_norm
    else:
        normed_grad = grad

    # weight factor and sign flip
    normed_grad *= -math.fabs(args.gan_toenc)
    return normed_grad