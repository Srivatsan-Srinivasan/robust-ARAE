######## NORMAL RUN WITH GRADIENT PENALTY ########################
python3 train.py --data_path snli_lm --bn_gen --bn_disc --tensorboard --cuda --tensorboard_freq 2000 --seed 45 --outf exp24 --tensorboard_logdir exp24 --timeit 50000 --gradient_penalty --arch_g 500-500 --eps_drift 0.001 --dropout 0.5 --min_epochs 3 --bidirectionnal --nhidden_dec 500

######## PROGRESSIVE VOCAB ###################################
python3 train.py --data_path snli_lm --bn_gen --bn_disc --tensorboard --cuda --tensorboard_freq 2000 --seed 45 --outf exp24 --tensorboard_logdir exp24 --timeit 50000 --gradient_penalty --arch_g 500-500 --eps_drift 0.001 --dropout 0.5 --min_epochs 3 --bidirectionnal --nhidden_dec 500 --progressive_vocab --vocabulary_switch_cutoff 0.85


