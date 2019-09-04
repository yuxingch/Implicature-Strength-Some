import argparse
import logging
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from utils import mkdir_p

logging.basicConfig(level=logging.INFO)


def split_train_test(seed_num, save_path, input='./some_fulldataset.csv'):
    logging.info('Spitting data into training/test sets\n========================')
    logging.info(f'Using random seed {seed_num}, file loaded from {input}')

    # set random seed
    random.seed(seed_num)
    # read in the file
    input_df = pd.read_csv(input, sep=',')
    dict_sentence_strength = input_df[['Item', 'StrengthSome']].groupby('Item')['StrengthSome'].apply(list).to_dict()
    dict_sentence_rating = input_df[['Item', 'Rating']].groupby('Item')['Rating'].apply(list).to_dict()
    dict_sentence_partitive = input_df[['Item', 'Partitive']].groupby('Item')['Partitive'].apply(list).to_dict()
    dict_sentence_modification = input_df[['Item', 'Modification']].groupby('Item')['Modification'].apply(list).to_dict()
    dict_sentence_subjecthood = input_df[['Item', 'BinaryGF']].groupby('Item')['BinaryGF'].apply(list).to_dict()
    list_new_value = []
    for (k, v) in dict_sentence_strength.items():
        values_strength = np.array(v)
        values_rating = np.array(dict_sentence_rating[k])
        is_partitive = 1 if dict_sentence_partitive[k][0] == 'yes' else 0
        is_modified = 1 if dict_sentence_modification[k][0] == 'modified' else 0
        print(dict_sentence_subjecthood[k][0])
        is_subject = 1 if dict_sentence_subjecthood[k][0] == 1 else 0
        l = k + ',' + format(np.mean(values_strength)) + ',' \
            + format(np.mean(values_rating)) + ',' + format(is_partitive) \
            + ',' + format(is_modified) + ',' + format(is_subject)
        list_new_value.append(l)
    # split
    total_num_examples = len(list_new_value)
    # shuffle
    ids = list(range(0, total_num_examples))
    random.shuffle(ids)
    train_ids = ids[:954]
    test_ids = ids[954:]
    mkdir_p(save_path)
    f = open(save_path + '/train_db.csv', 'w')
    head_line = "Item,StrengthSome,Rating,Partitive,Modification,Subjecthood\n"
    f.write(head_line)
    for i in train_ids:
        f.write(list_new_value[i]+"\n")
    f.close()
    f = open(save_path + '/test_db.csv', 'w')
    f.write(head_line)
    for i in test_ids:
        f.write(list_new_value[i]+"\n")
    f.close()
    return


def k_folds_idx(k, total_examples, seed_num):
    all_inds = list(range(total_examples))
    cv = KFold(n_splits=k, shuffle=True, random_state=seed_num)
    return cv.split(all_inds)
