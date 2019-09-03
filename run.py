import argparse
from collections import defaultdict
from datetime import datetime
import logging
import os
import pprint
import random
import re
from statistics import mean
import sys

from allennlp.commands.elmo import ElmoEmbedder
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import yaml

from models import split_by_whitespace, RatingModel, get_sentence, get_sentence_2d
from models import get_sentence_elmo
from models import get_sentence_bert
from models import parse_paragraph_2, parse_paragraph_3
from utils import mkdir_p


logging.basicConfig(level=logging.INFO)

cfg = edict()
cfg.SOME_DATABASE = './some_database.csv'
cfg.CONFIG_NAME = ''
cfg.RESUME_DIR = ''
cfg.SEED = 0
cfg.MODE = 'train'
cfg.PREDICTION_TYPE = 'rating'
cfg.IS_RANDOM = False
cfg.SINGLE_SENTENCE = True
cfg.EXPERIMENT_NAME = ''
cfg.GLOVE_DIM = 100
cfg.IS_ELMO = True
cfg.IS_BERT = False
cfg.ELMO_MODE = 'concat'
cfg.SAVE_PREDS = False
cfg.BATCH_ITEM_NUM = 29
cfg.PREDON = 'eval'
cfg.CUDA = False
cfg.GPU_NUM = 1

cfg.LSTM = edict()
cfg.LSTM.FLAG = False
cfg.LSTM.SEQ_LEN = 20
cfg.LSTM.HIDDEN_DIM = 200
cfg.LSTM.DROP_PROB = 0.2
cfg.LSTM.LAYERS = 2
cfg.LSTM.BIDIRECTION = True
cfg.LSTM.ATTN = False

# Training options
cfg.TRAIN = edict()
cfg.TRAIN.FLAG = True
cfg.TRAIN.BATCH_SIZE = 32
cfg.TRAIN.TOTAL_EPOCH = 200
cfg.TRAIN.INTERVAL = 4
cfg.TRAIN.START_EPOCH = 0
cfg.TRAIN.LR_DECAY_EPOCH = 20
cfg.TRAIN.LR = 5e-2
cfg.TRAIN.COEFF = edict()
cfg.TRAIN.COEFF.BETA_1 = 0.9
cfg.TRAIN.COEFF.BETA_2 = 0.999
cfg.TRAIN.COEFF.EPS = 1e-8
cfg.TRAIN.LR_DECAY_RATE = 0.8
cfg.TRAIN.DROPOUT = edict()
cfg.TRAIN.DROPOUT.FC_1 = 0.75
cfg.TRAIN.DROPOUT.FC_2 = 0.75

GLOVE_DIM = 100
NOT_EXIST = torch.FloatTensor(1, GLOVE_DIM).zero_()


def merge_yaml(new_cfg, old_cfg):
    for k, v in new_cfg.items():
        # check type
        old_type = type(old_cfg[k])
        if old_type is not type(v):
            if isinstance(old_cfg[k], np.ndarray):
                v = np.array(v, dtype=old_cfg[k].dtype)
            else:
                raise ValueError(('Type mismatch for config key: {}').format(k))
        # recursively merge dicts
        if type(v) is edict:
            try:
                merge_yaml(new_cfg[k], old_cfg[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            old_cfg[k] = v


def cfg_setup(filename):
    with open(filename, 'r') as f:
        new_cfg = edict(yaml.load(f))
    merge_yaml(new_cfg, cfg)


def load_dataset(input0, input1, input2, t):
    input_df0 = pd.read_csv(input0, sep='\t')
    input_df1 = pd.read_csv(input1, sep=',')
    input_df2 = pd.read_csv(input2, sep='\t')
    dict_item_sentence_raw = input_df0[['Item', 'Sentence']].drop_duplicates().groupby('Item')['Sentence'].apply(list).to_dict()
    dict_item_paragraph_raw = input_df2[['Item_ID', '20-b']].groupby('Item_ID')['20-b'].apply(list).to_dict()
    if t == 'strength':
        dict_item_mean_score_raw = input_df1[['Item', 'StrengthSome']].groupby('Item')['StrengthSome'].apply(list).to_dict()
    else:
        dict_item_mean_score_raw = input_df1[['Item', 'Rating']].groupby('Item')['Rating'].apply(list).to_dict()
    dict_item_mean_score = dict()
    dict_item_sentence = dict()
    dict_item_paragraph = dict()
    for (k, v) in dict_item_mean_score_raw.items():
        dict_item_mean_score[k] = v[0]
        dict_item_sentence[k] = dict_item_sentence_raw[k]
        dict_item_paragraph[k] = dict_item_paragraph_raw[k]
    return dict_item_mean_score, dict_item_sentence, dict_item_paragraph


def random_input(num_examples):
    res = []
    for i in range(num_examples):
        lst = []
        for j in range(GLOVE_DIM):
            lst.append(round(random.uniform(-1, 1), 16))
        res.append(lst)
    return torch.Tensor(res)


def main():
    embedder = ElmoEmbedder()
    parser = argparse.ArgumentParser(
        description='Run ...')
    parser.add_argument('--seed', dest='seed_nm', default=0, type=int)
    parser.add_argument('--mode', dest='mode', default='train')
    parser.add_argument('--t', dest='t', default='rating')
    parser.add_argument('--random', dest='random_vector', default=False)
    parser.add_argument('--save_preds', dest='save_preds', default=1)
    parser.add_argument('--name', dest='experiment_name', default="")
    parser.add_argument('--conf', dest='config_file', default="unspecified")
    opt = parser.parse_args()
    print(opt)

    if opt.config_file is not "unspecified":
        cfg_setup(opt.config_file)
        if not cfg.MODE == 'train':
            cfg.TRAIN.FLAG = False
    else:
        cfg.SEED = opt.seed_nm
        cfg.PREDICTION_TYPE = opt.t
        cfg.IS_RANDOM = opt.random_vector
        cfg.SAVE_PREDS = opt.save_preds
        cfg.EXPERIMENT_NAME = opt.experiment_name
        if not opt.mode == 'train':
            cfg.TRAIN.FLAG = False
            cfg.MODE = opt.mode

    # random seed
    random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(cfg.SEED)

    # print('Using configurations:')
    # pprint.pprint(cfg)

    curr_path = "./datasets/seed_" + str(cfg.SEED)
    if cfg.EXPERIMENT_NAME == "":
        cfg.EXPERIMENT_NAME = datetime.now().strftime('%m_%d_%H_%M')
    log_path = os.path.join(cfg.EXPERIMENT_NAME, "Logging")
    mkdir_p(log_path)
    file_handler = logging.FileHandler(os.path.join(log_path, cfg.MODE+"_log.txt"))
    logging.getLogger().addHandler(file_handler)

    logging.info('Using configurations:')
    logging.info(pprint.pformat(cfg))

    if cfg.MODE == 'train':
        load_db = curr_path + "/train_db.csv"
    elif cfg.MODE == 'eval':
        load_db = curr_path + "/" + cfg.PREDON + "_db.csv"
    elif cfg.MODE == 'all':
        load_db = curr_path + "/all_db.csv"
    labels, target_utterances, contexts = load_dataset(cfg.SOME_DATABASE,
                                                       load_db,
                                                       "./swbdext.csv",
                                                       cfg.PREDICTION_TYPE)

    curr_max = 7
    curr_min = 1
    original_labels = []

    # normalize the values [1,7] --> [0,1]
    normalized_labels = []
    max_diff = curr_max - curr_min
    keys = []
    for (k, v) in labels.items():
        keys.append(k)
        original_labels.append(float(v))
        labels[k] = (float(v) - curr_min) / max_diff
        normalized_labels.append(labels[k])
    cfg.BATCH_ITEM_NUM = len(normalized_labels)//cfg.TRAIN.BATCH_SIZE

    # obtain pre-trained word vectors
    word_embs = []
    word_embs_np = None
    word_embs_stack = None

    NUMPY_DIR = './datasets/seed_' + str(cfg.SEED)
    # is contextual or not
    if not cfg.SINGLE_SENTENCE:
        NUMPY_DIR += '_contextual'
    # type of pre-trained word embedding
    if cfg.IS_ELMO:
        NUMPY_DIR = '/elmo_' + cfg.ELMO_MODE
    elif cfg.IS_BERT:
        NUMPY_DIR += '/bert'
    else:  # default: GloVe
        NUMPY_DIR += './glove'
    # Avg/LSTM
    if cfg.LSTM.FLAG:
        NUMPY_DIR += '_lstm'
        NUMPY_PATH = NUMPY_DIR + '/embs_' + cfg.PREDON + '_' + format(cfg.LSTM.SEQ_LEN) + '.npy'
        LENGTH_PATH = NUMPY_DIR + '/len_' + cfg.PREDON + '_' + format(cfg.LSTM.SEQ_LEN) + '.npy'
    else:
        NUMPY_PATH = NUMPY_DIR + '/embs_' + cfg.PREDON + '.npy'
        LENGTH_PATH = NUMPY_DIR + '/len_' + cfg.PREDON + '.npy'
    mkdir_p(NUMPY_DIR)
    print(NUMPY_PATH)
    logging.info(f'Path to the current word embeddings: {NUMPY_PATH}')
    if os.path.isfile(NUMPY_PATH):
        word_embs_np = np.load(NUMPY_PATH)
        len_np = np.load(LENGTH_PATH)
        sl = len_np.tolist()
        word_embs_stack = torch.from_numpy(word_embs_np)
    else:
        sl = []
        for (k, v) in tqdm(target_utterances.items(), total=len(target_utterances)):
            if cfg.SINGLE_SENTENCE:
                # only including the target utterance
                input_text = v[0]
            else:
                # discourse context + target utterance
                context_v = contexts[k]
                input_text = context_v[0] + v[0]
            if cfg.IS_ELMO:
                curr_emb, l = get_sentence_elmo(input_text, embedder=embedder,
                                                elmo_mode=cfg.ELMO_MODE,
                                                not_contextual=cfg.SINGLE_SENTENCE,
                                                LSTM=cfg.LSTM.FLAG,
                                                seq_len=cfg.LSTM.SEQ_LEN)
            elif cfg.IS_BERT:
                curr_emb, l = get_sentence_bert(input_text, LSTM=cfg.LSTM.FLAG,
                                                max_seq_len=cfg.LSTM.SEQ_LEN,
                                                is_single=cfg.SINGLE_SENTENCE)
            else:
                curr_emb, l = get_sentence(input_text, seq_len=cfg.LSTM.SEQ_LEN)
            sl.append(l)
            word_embs.append(curr_emb)
        np.save(LENGTH_PATH, np.array(sl))
        word_embs_stack = torch.stack(word_embs)
        np.save(NUMPY_PATH, word_embs_stack.numpy())

    # If want to experiment with random embeddings:
    fake_embs = None
    if opt.random_vector:
        print("randomized word vectors")
        if cfg.TRAIN.FLAG:
            fake_embs = random_input(954)
        else:
            fake_embs = random_input(408)

    cfg.BATCH_ITEM_NUM = len(sl) // cfg.TRAIN.BATCH_SIZE

    if cfg.TRAIN.FLAG:
        save_path = "./" + cfg.EXPERIMENT_NAME + "_" + cfg.PREDICTION_TYPE + "_" + str(cfg.SEED)
        if cfg.IS_RANDOM:
            save_path += "_random"
            r_model = RatingModel(cfg, save_path)
            r_model.train(fake_embs, np.array(normalized_labels))
        else:
            r_model = RatingModel(cfg, save_path)
            r_model.train(word_embs_stack.float(), np.array(normalized_labels), sl)
    else:
        eval_path = "./" + cfg.EXPERIMENT_NAME + "_" + cfg.PREDICTION_TYPE + "_" + str(cfg.SEED)
        epoch_lst = [0, 1]
        i = 0
        while i < cfg.TRAIN.TOTAL_EPOCH - cfg.TRAIN.INTERVAL + 1:
            i += cfg.TRAIN.INTERVAL
            epoch_lst.append(i)
        logging.info(f'epochs to eval: {epoch_lst}')
        if cfg.IS_RANDOM:
            eval_path += "_random"
            load_path = eval_path + "/Model"
            for epoch in epoch_lst:
                cfg.RESUME_DIR = load_path + "/RNet_epoch_" + format(epoch)+".pth"
                eval_model = RatingModel(cfg, eval_path)
                preds = eval_model.evaluate(fake_embs, max_diff, curr_min, sl)
        else:
            load_path = eval_path + "/Model"
            max_epoch_dir = None
            max_value = -1.0
            max_epoch = None
            curr_coeff_lst = []
            for epoch in epoch_lst:
                cfg.RESUME_DIR = load_path + "/RNet_epoch_" + format(epoch)+ ".pth"
                eval_model = RatingModel(cfg, eval_path)
                preds, attn_weights = eval_model.evaluate(word_embs_stack.float(), max_diff, curr_min, sl)

                if cfg.LSTM.ATTN:
                    attn_path = eval_path + '/Attention'
                    mkdir_p(attn_path)
                    new_file_name = attn_path + '/' + cfg.PREDON + '_attn_epoch' + format(epoch) + '.npy'
                    np.save(new_file_name, attn_weights)
                    logging.info(f'Write attention weights to {new_file_name}.')

                curr_coeff = np.corrcoef(preds, np.array(original_labels))[0, 1]
                curr_coeff_lst.append(curr_coeff)
                if max_value < curr_coeff:
                    max_value = curr_coeff
                    max_epoch_dir = cfg.RESUME_DIR
                    max_epoch = epoch
                if cfg.SAVE_PREDS:
                    pred_file_path = eval_path + '/Preds'
                    mkdir_p(pred_file_path)
                    new_file_name = pred_file_path + '/' + cfg.PREDON + '_preds_rating_epoch' + format(epoch) + '.csv'
                    f = open(new_file_name, 'w')
                    head_line = "Item_ID\toriginal_mean\tpredicted\n"
                    print(f'Start writing predictions to file:\n{new_file_name}\n...')
                    f.write(head_line)
                    for i in range(len(keys)):
                        k = keys[i]
                        ori = original_labels[i]
                        pre = preds[i]
                        curr_line = k + '\t' + format(ori) + '\t' + format(pre)
                        f.write(curr_line+"\n")
                    f.close()
            logging.info(f'Max r = {max_value} achieved at epoch {max_epoch}')
            print(curr_coeff_lst)
    return

if __name__ == "__main__":
    main()
