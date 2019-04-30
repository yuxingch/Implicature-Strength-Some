# Implicature-Strength-Rating
`RatingModel` used to predict the implicature strength rating for *some (but not all)*.

## Latest Version
`0.0.4`: enable config file to keep track of experiment settings

## Installation
To set up the virtual environment (python3) to run the script:
```
sudo pip install virtualenv        # You will need to do this only once
virtualenv -p python3 .env         # Creates a virtual environment with python3
source .env/bin/activate           # Activate the virtual environment
pip install -r requirements.txt    # Install all the dependencies
deactivate                         # Exit the virtual environment when you're done
```

## Preprocessing Dataset
Splitting data into training/evaluation sets: (70%/30%)
```
python split_dataset.py --seed=SEED --input=PATH/TO/DATASET
```

Or, can use the default setting by running:
```
python split_dataset.py
```
The path to the training set will be `./datasets/seed_SEED/train_db.csv`. 

The path to the evaluation set will be `./datasets/seed_SEED/eval_db.csv`.

## Example
In `./cfg/`, there is one example configuration file (`./cfg/test_conf.yml`)
for training and there is one example configuration file (`./cfg/test_eval_conf.yml`) for evaluation. Using configuration files helps us simplify the process of feeding the experiment settings into the model. All tunable parameters are stored in `run.py` as `cfg`. 
```
cfg.SOME_DATABASE = './some_database.csv' # where we load the dataset
cfg.CONFIG_NAME = ''                      # name for this configuration
cfg.RESUME_DIR = ''                       # path to the previous checkpoint
cfg.SEED = 0                              # setting the random seed
cfg.MODE = 'train'                        # train/analyze/eval, default in train mode
cfg.PREDICTION_TYPE = 'rating'            # rating/strength, default: predict the implicature strength rating
cfg.IS_RANDOM = False                     # use random vectors to represent sentences, default False
cfg.SINGLE_SENTENCE = True                # only use the target sentence
cfg.EXPERIMENT_NAME = ''                  # name for this experiment
cfg.GLOVE_DIM = 100                       # GloVe dimension
cfg.IS_ELMO = True                        # use ELMo
cfg.ELMO_MODE = 'concat'                  # avg/concat, take the average of the ELMo vectors/concatenate the vectors
cfg.SAVE_PREDS = False                    # save the predictions as .csv file
cfg.BATCH_ITEM_NUM = 29                   # number of items in each batch

cfg.LSTM = edict()
cfg.LSTM.FLAG = False                     # whether using LSTM encoder or not
cfg.LSTM.SEQ_LEN = 20                     # the maximum sentence length, including `'<bos>'` and `'<eos>'`
cfg.LSTM.HIDDEN_DIM = 512                 # LSTM hidden dimension
cfg.LSTM.DROP_PROB = 0.2                  # LSTM drop probability, when number of layers is > 1
cfg.LSTM.LAYERS = 2                       # number of LSTM layers
cfg.LSTM.BIDIRECTION = True               # use bidirectional LSTM or not

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

User can also use the command-line arguments to specify some of these values:
- `--seed`: same as `cfg.SEED`
- `--mode`: same as `cfg.MODE`
- `--t`: same as `cfg.PREDICTION_TYPE`
- `--random`: same as `cfg.IS_RANDOM`
- `--sn`: number of sentences we take into consideration for each example. If `sn==0`, then it is equivalent to `cfg.SINGLE_SENTENCE == True`
- `--save_preds`: same as `cfg.SAVE_PREDS`
- `--name`: same as `cfg.EXPERIMENT_NAME`
- `--conf`: the path to the configuration file, if required

To use new parameters coming from the configuration file (e.g. `my_conf.yml`), first make sure the file is in `./cfg/`. Then use this line to run the script:
```
python run.py --conf='./cfg/my_conf.yml'
```

To visualize the loss in TensorBoard, run:
```
tensorboard --logdir path/to/logs
```
