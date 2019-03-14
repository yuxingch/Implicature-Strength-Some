import argparse
import random

import numpy as np
import pandas as pd

from utils import mkdir_p


def main():
    parser = argparse.ArgumentParser(
        description='Spitting data into training/evaluation sets ...')
    parser.add_argument('--seed', dest='seed_num', default=0, type=int)
    parser.add_argument('--input', dest='input', default='./some_fulldataset.csv')
    opt = parser.parse_args()
    print(opt)

    # set random seed
    random.seed(opt.seed_num)
    # read in the file
    input_df = pd.read_csv(opt.input, sep=',')
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
    eval_ids = ids[954:]
    path_nm = './datasets/seed_' + str(opt.seed_num)
    mkdir_p(path_nm)
    f = open(path_nm+'/train_db.csv', 'w')
    head_line = "Item,StrengthSome,Rating,Partitive,Modification,Subjecthood\n"
    f.write(head_line)
    for i in train_ids:
        f.write(list_new_value[i]+"\n")
    f.close()
    f = open(path_nm+'/eval_db.csv', 'w')
    f.write(head_line)
    for i in eval_ids:
        f.write(list_new_value[i]+"\n")
    f.close()
    return

if __name__ == '__main__':
    main()
