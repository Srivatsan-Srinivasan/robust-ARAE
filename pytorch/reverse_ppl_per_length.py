import argparse
import numpy as np
from utils import variable
import os


def init_config():
    """Just to make it more readable in PyCharm"""
    parser = argparse.ArgumentParser(description='Generate a dataset with a given model')
    parser.add_argument('--outf', type=str, required=True,
                        help="It will be a file located in the folder snli_lm_synthetic/")
    parser.add_argument('--gpu_id', type=int, default=None)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--norm_penalty', action='store_true',
                        help='Whether you want to load a model that was trained using norm-penalty'
                             'If that is the case, you need to indicate its ppl as well')
    parser.add_argument('--model', type=str, required=True,
                        help='Example: exp23ddd, exp22, ...'
                             'This model should be located in `output/`')
    parser.add_argument('--ppl', type=str, default=None,
                        help="If you want to load the model using norm-penalty, you need to"
                             "indicate what ppl the model has (it is indicated in the filename .pt)")
    args = parser.parse_args()
    return args


assert 'snli_lm_synthetic' in os.listdir('./'), "You need to create a folder snli_lm_synthetic first"


args = init_config()
if not args.norm_penalty:
    from models import load_models
    model_args, word2idx, idx2word, autoencoder, gan_gen, gan_disc = load_models('output/' + args.model)
else:
    assert args.ppl is not None, "If you load a model that was trained using norm-penalty, you need to " \
                                 "indicate the ppl it had (it is written in the filename)"
    from models import Seq2Seq, MLP_D, MLP_G
    import os
    import torch as t
    import json
    def load_models(load_path, ppl, old=False):
        model_args = json.load(open("{}/args.json".format(load_path), "r"))
        word2idx = json.load(open("{}/vocab.json".format(load_path), "r"))
        idx2word = {v: k for k, v in word2idx.items()}

        autoencoder = Seq2Seq(emsize=model_args['emsize'],
                              nhidden_enc=model_args['nhidden_enc'] if not old else model_args['nhidden'],
                              nhidden_dec=model_args['nhidden_dec'] if not old else model_args['nhidden'],
                              ntokens=model_args['ntokens'],
                              nlayers=model_args['nlayers'],
                              hidden_init=model_args['hidden_init'],
                              norm_penalty=model_args['norm_penalty'],
                              norm_penalty_threshold=model_args['norm_penalty_threshold'],
                              dropout=model_args['dropout'],
                              bidirectionnal=model_args['bidirectionnal']
                              )
        gan_gen = MLP_G(ninput=model_args['z_size'],
                        noutput=model_args['nhidden_enc'] if not old else model_args['nhidden'],
                        layers=model_args['arch_g'],
                        batchnorm=model_args['bn_gen']
                        )
        gan_disc = MLP_D(ninput=model_args['nhidden_enc'] if not old else model_args['nhidden'],
                         noutput=1,
                         layers=model_args['arch_d'],
                         spectralnorm=model_args['spectralnorm'],
                         batchnorm=model_args['bn_disc'],
                         std_minibatch=model_args['std_minibatch']
                         )

        print('Loading models from' + load_path)
        ae_path = os.path.join(load_path, "autoencoder_model_{}.pt".format(ppl))
        gen_path = os.path.join(load_path, "gan_gen_model_{}.pt".format(ppl))
        disc_path = os.path.join(load_path, "gan_disc_model_{}.pt".format(ppl))

        autoencoder.load_state_dict(t.load(ae_path))
        gan_gen.load_state_dict(t.load(gen_path))
        gan_disc.load_state_dict(t.load(disc_path))
        return model_args, word2idx, idx2word, autoencoder, gan_gen, gan_disc
    ppl = args.ppl
    model_args, word2idx, idx2word, autoencoder, gan_gen, gan_disc = load_models('output/'+args.model, 'intermediate_ppl_' + str(ppl))


from collections import Counter
counter = Counter()
ready = False


outf = 'snli_lm_synthetic/' + args.outf
i = 0
while not ready:
    i += 1
    z = variable(np.random.normal(size=(args.batch_size, 100)), cuda=True, gpu_id=args.gpu_id)
    code = gan_gen(z)
    sentences = autoencoder.generate(code, 30).squeeze(2)
    all_text = []
    for s in sentences.cpu().data.numpy():
        text = ''
        count = 0
        for idx in s:
            word = idx2word[idx]
            if word == '<eos>':
                break
            text += word
            count += 1
            text += ' '
        if count < 5:
            continue
        if counter[count] < int(1000000/count) and 5 <= count <= 24:
            counter[count] += 1
            all_text.append(text)
    ready = (set([v >= int(1000000/k) for k,v in counter.items()]) == {True})
    with open(outf, 'a') as f:
        f.write('\n'.join(all_text))
        f.write('\n')
