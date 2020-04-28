# Harnessing the linguistic signal to predict scalar inferences
In this repository, we include the PyTorch implementation for predicting the implicature strength rating for *some (but not all)* as discussed in [this paper](https://arxiv.org/pdf/1910.14254.pdf).

# Latest Version
- Initial edits for cleaning-up the repository

# Installation
Clone the repository and `cd` into the directory:
```
git clone https://github.com/yuxingch/Implicature-Strength-Some.git
cd Implicature-Strength-Some
```
To set up the virtual environment (python3) to run the script:
```
sudo pip install virtualenv        # You will need to do this only once
virtualenv -p python3 .env         # Creates a virtual environment with python3
source .env/bin/activate           # Activate the virtual environment
pip install -r requirements.txt    # Install all the dependencies
deactivate                         # Exit the virtual environment when you're done
```
Save `some_database.csv`, `some_fulldataset.csv`, and `swbdext.csv` in `./corpus_data/` directory:
```
- Implicature-Strength-Some
    - code
        - ...
    - corpus_data
        - some_database.csv
        - some_fulldataset.csv
        - swbdext.csv
```

# Run Experiments
## Preprocessing Dataset (Not required to run separately)
We don't have to run this separately. In the latest version, this has already been handled in `run.py`.

Splitting data into training/test sets directly (by default 70%/30%):
```
python ./code/split_dataset.py --seed=SEED_NUM --path=SAVE/PATH  --ratio=SPLIT_RATIO  --file=PATH/TO/CORPUS --verbose
```

Or, can use the default setting by running:
```
python ./code/split_dataset.py
```
Then the path to the training set will be `./datasets/train_db.csv` and the path to the test set will be `./datasets/test_db.csv`. We also have a `./datasets/all_db.csv` that combines the previous two files.

Sample output with default settings (if verbose):
```
Spit data into training/test sets with split ratio=0.7
=====================
Using random seed 0, file loaded from ./corpus_data/some_fulldataset.csv
New files can be found in this directory: ./datasets
Out of total 1362 entries, 954 will be in training set and 408 will be in test set.
=====================
```

In actual runs, based on the implementation in `run.py`, the path will be `./{data_path}/seed_{cfg.SEED}/{train/test/all}_db.csv`

## Configuration File
In `./cfg/`, there are example configuration files (e.g. `./cfg/cv_elmo_lstm_attn_context.yml`)
for our experiments. Using configuration files helps us simplify the process of feeding the experiment settings into the model. All tunable parameters are stored in `run.py` as `cfg`. 
```
cfg.SOME_DATABASE = './some_database.csv' # where we load the dataset
cfg.CONFIG_NAME = ''                      # configuration name
cfg.RESUME_DIR = ''                       # path to the previous checkpoint we want to resume
cfg.SEED = 0                              # set random seed, default: 0
cfg.MODE = 'train'                        # train/test, default: train mode
cfg.PREDICTION_TYPE = 'rating'            # rating/strength, default: predict the implicature strength rating
cfg.MAX_VALUE = 7                         # max value in our raw data
cfg.MIN_VALUE = 1                         # min value in our raw data
cfg.IS_RANDOM = False                     # use random vectors to represent sentences, default: False
cfg.SINGLE_SENTENCE = True                # only use the target utterance
cfg.EXPERIMENT_NAME = ''                  # experiment name
cfg.OUT_PATH = './'                       # where we store the output (log, models, and etc.)
cfg.GLOVE_DIM = 100                       # GloVe dimension
cfg.IS_ELMO = True                        # use ELMo
cfg.IS_BERT = False                       # use BERT
cfg.ELMO_LAYER = 2
cfg.BERT_LAYER = 11
cfg.BERT_LARGE = False
cfg.ELMO_MODE = 'concat'                  # avg/concat, take the average of the ELMo vectors/concatenate the vectors
cfg.SAVE_PREDS = False                    # save the predictions as .csv file (in test mode only)
cfg.BATCH_ITEM_NUM = 30                   # number of examples in each batch
cfg.PREDON = 'test'                       # which set we want to make predictions on: train/test
cfg.CUDA = False                          # use GPU or not, default: False
cfg.GPU_NUM = 1                           # number of GPUs we use, default: 1
cfg.KFOLDS = 5                            # number of folds, default: 5
cfg.CROSS_VALIDATION_FLAG = True          # train with cross validation, default: True
cfg.SPLIT_NAME = ""

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

cfg.EVAL = edict()
cfg.EVAL.FLAG = False
cfg.EVAL.BEST_EPOCH = 100
```

We can use the command-line argument to specify the path to the configuration file (see next section).

## Start training
To use new parameters coming from the configuration file (e.g. `./cfg/cv_elmo_lstm_attn_context.yml`), first make sure the file is in `./cfg/`. Then use this line to run the script:
```
python ./code/run.py --conf='./cfg/cv_elmo_lstm_attn_context.yml'
```
The outputs will be stored in this hierachy (some examples):
```
- Implicature_Strength_Some
    - bert_lstm_attn
        - ...
    - code
    - corpus_data
    - datasets
        - seed_0
            - bert_layer_11_lstm
                - embs_train_30.npy
                - len_train_30.npy
            - elmo_layer_2_lstm
                - embs_train_30.npy
                - len_train_30.npy 
            - glove_lstm
                - embs_train_30.npy
                - len_train_30.npy
            - all_db.csv
            - test_db.csv
            - train_db.csv
        - ...
    - elmo_lstm_attn
        - 1
            - Best Model
                - RNet_epoch_{the_best_x}.pth
            - Model
                - RNet_epoch_{x}.pth
        - ...
        - Logging
            -  train_log.txt
    - glove_lstm_attn
        - ...
```

If you use these models, please cite the following paper:
```
@article{schuster2019harnessing,
  title={Harnessing the linguistic signal to predict scalar inferences},
  author={Schuster, Sebastian and Chen, Yuxing and Degen, Judith},
  journal={arXiv preprint arXiv:1910.14254},
  year={2019}
}
```
