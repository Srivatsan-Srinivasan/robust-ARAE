import os
from itertools import product


experiments = [
    "python3 main.py --model NNLM2 --l2_penalty 0.0001 --batch_size 320 --optimizer Adam --lr 0.001 --emb_train True --early_stopping True --save True --output_filename nnlm2/0",
    "python3 main.py --model NNLM2 --l2_penalty 0.0001 --batch_size 320 --optimizer Adam --lr 0.001 --emb_train True --early_stopping True --save True --output_filename nnlm2/1",
    "python3 main.py --model NNLM2 --l2_penalty 0.0001 --batch_size 320 --optimizer Adam --lr 0.001 --emb_train True --early_stopping True --save True --output_filename nnlm2/2",
    "python3 main.py --model NNLM2 --l2_penalty 0.0001 --batch_size 320 --optimizer Adam --lr 0.001 --emb_train True --early_stopping True --save True --output_filename nnlm2/3",
    "python3 main.py --model NNLM2 --l2_penalty 0.0001 --batch_size 320 --optimizer Adam --lr 0.001 --emb_train True --early_stopping True --save True --output_filename nnlm2/4",
    "python3 main.py --model NNLM2 --l2_penalty 0.0001 --batch_size 320 --optimizer Adam --lr 0.001 --emb_train True --early_stopping True --save True --output_filename nnlm2/5",
    "python3 main.py --model NNLM2 --l2_penalty 0.0001 --batch_size 320 --optimizer Adam --lr 0.001 --emb_train True --early_stopping True --save True --output_filename nnlm2/6",
    "python3 main.py --model NNLM2 --l2_penalty 0.0001 --batch_size 320 --optimizer Adam --lr 0.001 --emb_train True --early_stopping True --save True --output_filename nnlm2/7",
    "python3 main.py --model NNLM2 --l2_penalty 0.0001 --batch_size 320 --optimizer Adam --lr 0.001 --emb_train True --early_stopping True --save True --output_filename nnlm2/8",
    "python3 main.py --model NNLM2 --l2_penalty 0.0001 --batch_size 320 --optimizer Adam --lr 0.001 --emb_train True --early_stopping True --save True --output_filename nnlm2/9",
    "python3 main.py --model NNLM2 --l2_penalty 0.0001 --batch_size 320 --optimizer Adam --lr 0.001 --emb_train True --early_stopping True --save True --output_filename nnlm2/10",
    "python3 main.py --model NNLM2 --l2_penalty 0.0001 --batch_size 320 --optimizer Adam --lr 0.001 --emb_train True --early_stopping True --save True --output_filename nnlm2/11",
    "python3 main.py --model NNLM2 --l2_penalty 0.0001 --batch_size 320 --optimizer Adam --lr 0.001 --emb_train True --early_stopping True --save True --output_filename nnlm2/12",
    "python3 main.py --model NNLM2 --l2_penalty 0.0001 --batch_size 320 --optimizer Adam --lr 0.001 --emb_train True --early_stopping True --save True --output_filename nnlm2/13",
    "python3 main.py --model NNLM2 --l2_penalty 0.0001 --batch_size 320 --optimizer Adam --lr 0.001 --emb_train True --early_stopping True --save True --output_filename nnlm2/14",
    "python3 main.py --model NNLM2 --l2_penalty 0.0001 --batch_size 320 --optimizer Adam --lr 0.001 --emb_train True --early_stopping True --save True --output_filename nnlm2/15",
    "python3 main.py --model NNLM2 --l2_penalty 0.0001 --batch_size 320 --optimizer Adam --lr 0.001 --emb_train True --early_stopping True --save True --output_filename nnlm2/16",
    "python3 main.py --model NNLM2 --l2_penalty 0.0001 --batch_size 320 --optimizer Adam --lr 0.001 --emb_train True --early_stopping True --save True --output_filename nnlm2/17",
    "python3 main.py --model NNLM2 --l2_penalty 0.0001 --batch_size 320 --optimizer Adam --lr 0.001 --emb_train True --early_stopping True --save True --output_filename nnlm2/18",
    "python3 main.py --model NNLM2 --l2_penalty 0.0001 --batch_size 320 --optimizer Adam --lr 0.001 --emb_train True --early_stopping True --save True --output_filename nnlm2/19",
    "python3 main.py --model NNLM2 --l2_penalty 0.0001 --batch_size 320 --optimizer Adam --lr 0.001 --emb_train True --early_stopping True --save True --output_filename nnlm2/20",
    "python3 main.py --model NNLM2 --l2_penalty 0.0001 --batch_size 320 --optimizer Adam --lr 0.001 --emb_train True --early_stopping True --save True --output_filename nnlm2/21",
    "python3 main.py --model NNLM2 --l2_penalty 0.0001 --batch_size 320 --optimizer Adam --lr 0.001 --emb_train True --early_stopping True --save True --output_filename nnlm2/22",
    "python3 main.py --model NNLM2 --l2_penalty 0.0001 --batch_size 320 --optimizer Adam --lr 0.001 --emb_train True --early_stopping True --save True --output_filename nnlm2/23",
    "python3 main.py --model NNLM2 --l2_penalty 0.0001 --batch_size 320 --optimizer Adam --lr 0.001 --emb_train True --early_stopping True --save True --output_filename nnlm2/24",
    "python3 main.py --model NNLM2 --l2_penalty 0.0001 --batch_size 320 --optimizer Adam --lr 0.001 --emb_train True --early_stopping True --save True --output_filename nnlm2/25",
    "python3 main.py --model NNLM2 --l2_penalty 0.0001 --batch_size 320 --optimizer Adam --lr 0.001 --emb_train True --early_stopping True --save True --output_filename nnlm2/26",
    "python3 main.py --model NNLM2 --l2_penalty 0.0001 --batch_size 320 --optimizer Adam --lr 0.001 --emb_train True --early_stopping True --save True --output_filename nnlm2/27",
    "python3 main.py --model NNLM2 --l2_penalty 0.0001 --batch_size 320 --optimizer Adam --lr 0.001 --emb_train True --early_stopping True --save True --output_filename nnlm2/28",
    "python3 main.py --model NNLM2 --l2_penalty 0.0001 --batch_size 320 --optimizer Adam --lr 0.001 --emb_train True --early_stopping True --save True --output_filename nnlm2/29",
    "python3 main.py --model NNLM2 --l2_penalty 0.0001 --batch_size 320 --optimizer Adam --lr 0.001 --emb_train True --early_stopping True --save True --output_filename nnlm2/30",
    "python3 main.py --model NNLM2 --l2_penalty 0.0001 --batch_size 320 --optimizer Adam --lr 0.001 --emb_train True --early_stopping True --save True --output_filename nnlm2/31",
]


con_size = [5, 8]
BN = [False, True]
dropout = [0, 0.5]
activation = ['gated', 'lrelu']
hdim = [50, 100]

for k, (c,b,d,a,h) in enumerate(product(con_size, BN, dropout, activation, hdim)):
    experiments[k] = experiments[k] + ' --con_size %s --dropout %s --BN %s --nnlm_hdim %s --nnlm_acti %s' % (c, str(d), str(b), str(h), a)

if 'nnlm2' not in os.listdir():
    os.system('mkdir nnlm2')
for experiment in experiments:
    os.system(experiment)
