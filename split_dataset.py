import argparse
import random

import numpy as np
import pandas as pd

from utils import mkdir_p


def main():
    parser = argparse.ArgumentParser(
        description='Spitting data into training/evaluation sets ...')
    parser.add_argument('--seed', dest='seed_num', default=0, type=int)
    parser.add_argument('--input', dest='input', default='./sentence_si_means.csv')
    opt = parser.parse_args()
    print(opt)

    # set random seed
    random.seed(opt.seed_num)
    # read in the file
    input_df = pd.read_csv(opt.input, sep=',')
    # dict_sentence_strength = input_df[['Item', 'StrengthSome']].groupby('Item')['StrengthSome'].apply(list).to_dict()
    # dict_sentence_rating = input_df[['Item', 'Rating']].groupby('Item')['Rating'].apply(list).to_dict()
    # dict_sentence_partitive = input_df[['Item', 'Partitive']].groupby('Item')['Partitive'].apply(list).to_dict()
    # dict_sentence_modification = input_df[['Item', 'Modification']].groupby('Item')['Modification'].apply(list).to_dict()
    # dict_sentence_subjecthood = input_df[['Item', 'BinaryGF']].groupby('Item')['BinaryGF'].apply(list).to_dict()
    dict_sentence_proportion = input_df[['tgrep.id', 'proportion_goodsentence']].groupby('tgrep.id')['proportion_goodsentence'].apply(list).to_dict()
    dict_sentence_bnb = input_df[['tgrep.id', 'Sentence_BNB']].groupby('tgrep.id')['Sentence_BNB'].apply(list).to_dict()
    dict_sentence_mean = input_df[['tgrep.id', 'Mean']].groupby('tgrep.id')['Mean'].apply(list).to_dict()
    dict_sentence_ymin = input_df[['tgrep.id', 'YMin']].groupby('tgrep.id')['YMin'].apply(list).to_dict()
    dict_sentence_ymax = input_df[['tgrep.id', 'YMax']].groupby('tgrep.id')['YMax'].apply(list).to_dict()
    list_new_value = []
    for (k, v) in dict_sentence_proportion.items():
        values_proportion = v[0]
        values_bnb = dict_sentence_bnb[k][0]
        values_mean = dict_sentence_mean[k][0]
        values_ymin = dict_sentence_ymin[k][0]
        values_ymax = dict_sentence_ymax[k][0]
        # values_rating = np.array(dict_sentence_rating[k])
        # is_partitive = 1 if dict_sentence_partitive[k][0] == 'yes' else 0
        # is_modified = 1 if dict_sentence_modification[k][0] == 'modified' else 0
        # print(dict_sentence_subjecthood[k][0])
        # is_subject = 1 if dict_sentence_subjecthood[k][0] == 1 else 0
        # l = k + ',' + format(np.mean(values_strength)) + ',' \
        #     + format(np.mean(values_rating)) + ',' + format(is_partitive) \
        #     + ',' + format(is_modified) + ',' + format(is_subject)
        l = k + ',' + format(values_proportion) + ',' + format(values_bnb) + ',' + format(values_mean) \
            + ',' + format(values_ymin) + ',' + format(values_ymax)
        list_new_value.append(l)
    # split
    total_num_examples = len(list_new_value)
    # shuffle
    ids = list(range(0, total_num_examples))
    random.shuffle(ids)
    train_ids = ids[:622]
    eval_ids = ids[622:]
    path_nm = './datasets/seed_' + str(opt.seed_num)
    mkdir_p(path_nm)
    f = open(path_nm+'/or_train_db.csv', 'w')
    head_line = "tgrep.id,proportion_goodsentence,Sentence_BNB,Mean,YMin,YMax\n"
    f.write(head_line)
    for i in train_ids:
        f.write(list_new_value[i]+"\n")
    f.close()
    f = open(path_nm+'/or_eval_db.csv', 'w')
    f.write(head_line)
    for i in eval_ids:
        f.write(list_new_value[i]+"\n")
    f.close()
    return

if __name__ == '__main__':
    main()
