from utils import get_ppl, train_ngram_lm
import argparse


def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tmp_file', type=str, required=True,
                        help="It will be a file located in the folder snli_lm_synthetic/")
    parser.add_argument('--source_file', type=str, required=True,
                        help="It will be a file located in the folder snli_lm_synthetic/")
    parser.add_argument('--outf', type=str, required=True)
    args = parser.parse_args()
    return args


def train_lm(length, filename, eval_path, save_path, kenlm_path):
    """Evaluate the performance of a simple language model (KENLM) that is trained on synthetic sentences"""
    # generate 100000 examples
    with open(save_path + '.txt', 'w') as new_f:
        with open(filename, 'r') as source_f:
            lines = list(filter(lambda x: len(x.split()) == length, source_f.read().split('\n')))
        new_f.write('\n'.join(lines[:int(100000 * 5 / length)]))

    # train language model on generated examples
    lm = train_ngram_lm(kenlm_path=kenlm_path,
                        data_path=save_path + ".txt",
                        output_path=save_path + ".arpa",
                        N=5)

    # load sentences to evaluate on
    with open(eval_path, 'r') as f:
        lines = f.readlines()
    sentences = [l.replace('\n', '') for l in lines]
    ppl = get_ppl(lm, sentences)

    return ppl


args = init_config()
results = []
for l in range(5, 24):
    results.append((l, train_lm(l, 'snli_lm_synthetic/'+args.source_file, 'snli_lm/test.txt', 'snli_lm_synthetic/'+args.tmp_file, '../Data/kenlm')))
with open('snli_lm_synthetic/'+args.outf, 'w') as f:
    f.write(args.source_file)
    f.write('\n')
    f.write('\n'.join([str(r) for r in results]))

