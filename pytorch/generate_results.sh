#snli train
#python train.py --data_path ./data_snli --cuda --kenlm_path kenlm

#yelp train
#python3 train.py --data_path ./data --batch_size 64 --maxlen 25 --vocab_size 30000 --lowercase --cuda --epoch 25

##################################################################
#########################START SCRIPT#############################
##################################################################

#1. generated text
#the data paths expect five files: 
#args.json, gan_disc_model.pt, vocab.json, autoencoder_model.pt, gan_gen_model.pt
cd pytorch
python generate.py --load_path ./maxlen30

#2. translations between texts
#args.json, gan_disc_model_25.pt, vocab.json, autoencoder_model_25.pt, gan_gen_model_25.pt
cd ../yelp
python3 transfer.py --data_path ./data --load_path ./example --cuda --epoch 25



#python3 transfer.py --data_path ./data --load_path ./example --cuda --epoch 25 --lm_path ../pytorch/kenlm
#python3 transfer.py --data_path ./data --load_path ./example --cuda --epoch 25 --lm_path ../pytorch/kenlm
