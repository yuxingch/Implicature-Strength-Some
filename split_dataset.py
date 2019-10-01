import argparse
import logging
import random
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from utils import mkdir_p

#logging.basicConfig(level=logging.INFO)


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

def split_k_fold(seed_num, save_path, splits=6, input='./some_fulldataset.csv'):
    logging.info(f'Spitting data into {splits} training/test splits\n========================')
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
    k_fraction = int(len(ids) / splits)
    for j in range(splits):
        train_ids = ids[0:k_fraction*j] + ids[k_fraction*(j+1):]
        test_ids = ids[k_fraction*j:k_fraction*(j+1)]
        split_save_path = os.path.join(save_path, str(j))
        mkdir_p(split_save_path)
        with open(split_save_path + '/train_db.csv', 'w') as f:
            head_line = "Item,StrengthSome,Rating,Partitive,Modification,Subjecthood\n"
            f.write(head_line)
            for i in train_ids:
                f.write(list_new_value[i]+"\n")
      
        with open(split_save_path + '/test_db.csv', 'w') as f:
            f.write(head_line)
            for i in test_ids:
                f.write(list_new_value[i]+"\n")
          

    return


def k_folds_idx(k, total_examples, seed_num):
    all_inds = list(range(total_examples))
    cv = KFold(n_splits=k, shuffle=True, random_state=seed_num)
    return cv.split(all_inds)
    
def main():
    parser = argparse.ArgumentParser(
        description='Creat data splits ...')
    parser.add_argument('--seed', dest='seed', type=int, default=0)
    parser.add_argument('--path', dest='path', type=str, required=True)
    parser.add_argument('-k', dest='k', type=int, default=6)
    opt = parser.parse_args()
    split_k_fold(opt.seed, opt.path, splits=opt.k)

if __name__ == '__main__':
    main()
    
    
