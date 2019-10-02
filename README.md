# Implicature-Strength-Rating
`RatingModel` used to predict the implicature strength rating for *some (but not all)*.

## Latest Version
- Able to run all models
- Add k-fold cross validation
- Sanity check ongoing

## Installation
To set up the virtual environment (python3) to run the script:
```
sudo pip install virtualenv        # You will need to do this only once
virtualenv -p python3 .env         # Creates a virtual environment with python3
source .env/bin/activate           # Activate the virtual environment
pip install -r requirements.txt    # Install all the dependencies
pip install allennlp               # Need to manually install allennlp
deactivate                         # Exit the virtual environment when you're done
```

## Preprocessing Dataset
We don't have to run this separately. In the latest version, this has already been handled in `run.py`.

Splitting data into training/test sets: (70%/30%)
```
python split_dataset.py --seed=SEED --input=PATH/TO/DATASET
```

Or, can use the default setting by running:
```
python split_dataset.py
```
The path to the training set will be `./datasets/seed_SEED/train_db.csv`. 

The path to the test set will be `./datasets/seed_SEED/test_db.csv`.

## Example
In `./cfg/`, there is one example configuration file (`./cfg/test_conf.yml`)
for training and there is one example configuration file (`./cfg/test_eval_conf.yml`) for evaluation. Using configuration files helps us simplify the process of feeding the experiment settings into the model. All tunable parameters are stored in `run.py` as `cfg`. 
```
cfg.SOME_DATABASE = './some_database.csv' # where we load the dataset
cfg.CONFIG_NAME = ''                      # configuration name
cfg.RESUME_DIR = ''                       # path to the previous checkpoint we want to resume
cfg.SEED = 0                              # set random seed, default: 0
cfg.MODE = 'train'                        # train/test, default: train mode
cfg.PREDICTION_TYPE = 'rating'            # rating/strength, default: predict the implicature strength rating
cfg.IS_RANDOM = False                     # use random vectors to represent sentences, default: False
cfg.SINGLE_SENTENCE = True                # only use the target utterance
cfg.EXPERIMENT_NAME = ''                  # experiment name
cfg.GLOVE_DIM = 100                       # GloVe dimension
cfg.IS_ELMO = True                        # use ELMo
cfg.ELMO_MODE = 'concat'                  # avg/concat, take the average of the ELMo vectors/concatenate the vectors
cfg.SAVE_PREDS = False                    # save the predictions as .csv file (in test mode only)
cfg.BATCH_ITEM_NUM = 30                   # number of examples in each batch
cfg.PREDON = 'test'                       # which set we want to make predictions on: train/test
cfg.CUDA = False                          # use GPU or not, default: False
cfg.GPU_NUM = 1                           # number of GPUs we use, default: 1
cfg.KFOLDS = 5                            # number of folds, default: 5
cfg.CROSS_VALIDATION_FLAG = True          # train with cross validation, default: True

cfg.LSTM = edict()
cfg.LSTM.FLAG = False                     # whether using LSTM encoder or not
cfg.LSTM.SEQ_LEN = 30                     # the maximum sentence length, including `'<bos>'` and `'<eos>'`, default: 30
cfg.LSTM.HIDDEN_DIM = 512                 # LSTM hidden dimension
cfg.LSTM.DROP_PROB = 0.2                  # LSTM drop probability, when number of layers is > 1
cfg.LSTM.LAYERS = 2                       # number of LSTM layers
cfg.LSTM.BIDIRECTION = True               # use bidirectional LSTM or not
cfg.LSTM.ATTN = False                     # with attention layer, default: False

# Training options
cfg.TRAIN = edict()
cfg.TRAIN.FLAG = True                     # True/False, whether we're in training mode
cfg.TRAIN.BATCH_SIZE = 32                 # batch size
cfg.TRAIN.TOTAL_EPOCH = 200               # total number of epochs to run
cfg.TRAIN.INTERVAL = 4                    # save the checkpoint for every _ epochs
cfg.TRAIN.START_EPOCH = 0                 # starting epoch
cfg.TRAIN.LR_DECAY_EPOCH = 20             # decrease the learning rate for every ### epochs
cfg.TRAIN.LR = 5e-2                       # intial learning rate
cfg.TRAIN.COEFF = edict()
cfg.TRAIN.COEFF.BETA_1 = 0.9              # coefficient for Adam optimizer
cfg.TRAIN.COEFF.BETA_2 = 0.999            # coefficient for Adam optimizer
cfg.TRAIN.COEFF.EPS = 1e-8                # coefficient for Adam optimizer
cfg.TRAIN.LR_DECAY_RATE = 0.8             # decay rate for the learning rate
cfg.TRAIN.DROPOUT = edict()
cfg.TRAIN.DROPOUT.FC_1 = 0.75             # drop out prob in fully connected layer 1
cfg.TRAIN.DROPOUT.FC_2 = 0.75             # drop out prob in fully connected layer 2
```

User can use the command-line argument to specify the path to the configuration file:
- `--conf`: the path to the configuration file, if required

## Getting Started
To start the BERT service:
```
bert-serving-start -model_dir BERT_model/uncased_L-12_H-768_A-12/ -num_worker=4 -max_seq_len=30 -pooling_strategy=NONE
```
This will start a service with four workers. The maximum length of sequence that a BERT model can handle is 30. By setting `pooling_strategy` to `None`, we get ELMo-like contextual word embeddings (i.e. the service will return a [30, 768] matrix for every sequence.).

To use new parameters coming from the configuration file (e.g. `my_conf.yml`), first make sure the file is in `./cfg/`. Then use this line to run the script:
```
python run.py --conf='./cfg/my_conf.yml'
```

To visualize the loss in TensorBoard, run:
```
tensorboard --logdir path/to/logs
```
