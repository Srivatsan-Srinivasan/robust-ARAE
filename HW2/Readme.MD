Entry Script : main.py - could be run from command line. For the list of arguments, check file. 

<b>CODE FLOW</b>

check_args -> Validates if there are any inconsistencies in the arguments

get_params -> Buckets the args into subgroup of params for different purposes. Both of them found in process_params.py

generate_iterators -> Loads data, splits them via torchtext iterators, loads TEXT vocab and embeddings. Found in data_process.py

train -> Trains the model and outputs the trained model. Found in train_seqmodels.py

predict -> Predicts from the trained model, calculates certain metrics and stores them if need be.

