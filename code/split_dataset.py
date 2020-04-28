import argparse
import logging
import math
import random
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from utils import mkdir_p


def split_train_test(seed_num, save_path, ratio=0.7, input='./corpus_data/some_fulldataset.csv',
                     verbose=True):
    """Split the corpus into training and test sets with a given split ratio

    Arguments:
    seed_num -- the random seed we want to use
    save_path -- where we store the new training/test files
    ratio -- split ratio
    input -- path to the corpus that we want to split
    verbose -- if true, will print message to screen
    """
    if verbose:
        print(f"Spit data into training/test sets with split ratio={ratio}\n=====================")
        print(f"Using random seed {seed_num}, file loaded from {input}")

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

    total_num_examples = len(dict_sentence_strength)
    train_num_examples = math.ceil(ratio*len(dict_sentence_strength))
    test_num_examples = total_num_examples - train_num_examples
    if verbose:
        print(f"New files can be found in this directory: {save_path}")
        print(f"Out of total {total_num_examples} entries, {train_num_examples} will be in training"
            + f" set and {test_num_examples} will be in test set.\n=====================")

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
    # sanity check
    assert total_num_examples == len(list_new_value)
    # shuffle
    ids = list(range(0, total_num_examples))
    random.shuffle(ids)
    train_ids = ids[:train_num_examples]
    test_ids = ids[train_num_examples:]
    mkdir_p(save_path)

    f_all = open(save_path + '/all_db.csv', 'w')
    f = open(save_path + '/train_db.csv', 'w')
    head_line = "Item,StrengthSome,Rating,Partitive,Modification,Subjecthood\n"
    f_all.write(head_line)
    f.write(head_line)
    for i in train_ids:
        f_all.write(list_new_value[i]+"\n")
        f.write(list_new_value[i]+"\n")
    f.close()
    f = open(save_path + '/test_db.csv', 'w')
    f.write(head_line)
    for i in test_ids:
        f_all.write(list_new_value[i]+"\n")
        f.write(list_new_value[i]+"\n")
    f_all.close()
    f.close()
    return


def k_folds_idx(k, total_examples, seed_num):
    """Create K folds

    Arguments:
    k -- number of folds we want
    total_examples -- total number of examples we want to split
    seed_num -- the random seed we want to use

    Return:
    output -- k (train_idx, val_idx) pairs
    """
    all_inds = list(range(total_examples))
    cv = KFold(n_splits=k, shuffle=True, random_state=seed_num)
    output = cv.split(all_inds)
    return output

def main():
    parser = argparse.ArgumentParser(
        description="Creating data splits ...")
    parser.add_argument("--seed", dest="seed", type=int, default=0)
    parser.add_argument("--path", dest="path", type=str, default="./datasets")
    parser.add_argument("--ratio", dest="ratio", type=float, default=0.7)
    parser.add_argument("--file", dest="input", type=str,
        default="./corpus_data/some_fulldataset.csv")
    parser.add_argument("--verbose", dest="verbose", action='store_true')
    opt = parser.parse_args()
    split_train_test(opt.seed, opt.path, opt.ratio, opt.input, opt.verbose)

if __name__ == '__main__':
    main()
