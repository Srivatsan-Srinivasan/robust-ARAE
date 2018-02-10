Entry Script : main.py - could be run from command line. For the list of arguments, check file. 

<b>CODE FLOW</b>

check\_args -> Validates if there are any inconsistencies in the arguments

get\_params -> Buckets the args into subgroup of params for different purposes. Both of them found in process\_params.py

generate\_iterators -> Loads data, splits them via torchtext iterators, loads TEXT vocab and embeddings. Found in data\_process.py

train -> Trains the model and outputs the trained model. Found in train\_seqmodels.py

predict -> Predicts from the trained model, calculates certain metrics and stores them if need be.

<b>IF YOU ADD A NEW MODEL</b>

EXISTING ALLOWED MODELS : LSTM, BiLSTM, GRU, BiGRU, NNLM, Trigram

1. Add your model class to lstm.py and ensure that you provide model argument in the command line with the same name as the class itself.

2. Check if you need any special processing of data into inputs and target outputs apart from what is done in train \_seqmodels.py

3. Add your model to const.py

