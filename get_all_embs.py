import re
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
import torchtext.vocab as vocab

GLOVE_DIM = 100

glove = vocab.GloVe(name='6B', dim=GLOVE_DIM)
# print('Loaded {} words'.format(len(glove.itos)))

torch.manual_seed(0)

_UNK = torch.randn(GLOVE_DIM,)
_PAD = torch.randn(GLOVE_DIM,)
NOT_EXIST = torch.FloatTensor(1, GLOVE_DIM).zero_()


_OOV = dict()


def get_word(w):
    try:
        result = glove.vectors[glove.stoi[w]]
    except KeyError:
        # result = _UNK
        if w in _OOV:
            result = _OOV[w]
        else:
            result = torch.randn(GLOVE_DIM,)
            _OOV[w] = result
        print(w)
    return result


def get_sentence(s, max_len=40):
    s = s.replace('\'ve', ' \'ve')
    s = s.replace('\'ll', ' \'ll')
    s = s.replace('n\'t', ' n\'t')
    s = s.replace('\'d', ' \'d')
    s = s.replace('-', ' ')
    s = s.replace('\'s', ' \'s')
    modified_s = re.sub('#', '.', s).strip('.').split('.')
    modified_s = list(filter(None, modified_s))
    raw_tokens = []
    for s in modified_s:
        s = re.sub('speaker[0-9a-z\-\*]*[0-9]', '', s)
        s = re.sub('[^a-zA-Z0-9- \n\.]', '', s)
        s = re.sub('n[0-9][0-9a-z]{4,5}', '', s)
        s = re.sub('[0-9]t[0-9]+', '', s)
        s = s.replace(' ve ', ' \'ve ')
        s = s.replace(' ll ', ' \'ll ')
        s = s.replace(' nt ', ' n\'t ')
        s = s.replace(' d ', ' \'d ')
        s = s.replace(' s ', ' \'s ')
        s = s.replace('doeuvres', 'd\'oeuvres')
        s = s.replace('mumblex', 'mumble')
        raw_tokens += split_by_whitespace(s)
    # s = re.sub('[^a-zA-Z0-9 \n\.]', '', s)
    # raw_tokens = split_by_whitespace(s)
    # print(raw_tokens)
    # n = len(raw_tokens)
    # if (n < max_len):
    #     lst = [get_word(w.lower()) for w in raw_tokens]
    #     # lst += [_PAD] * (max_len - n)
    # else:
    #     raw_tokens = raw_tokens[:max_len]
    #     lst = [get_word(w.lower()) for w in raw_tokens]
    lst = [get_word(w.lower()) for w in raw_tokens]
    all_embs = torch.stack(lst)
    return torch.mean(all_embs, 0), raw_tokens  # embedding_size


def split_by_whitespace(sentence):
    words = []
    for space_separated_fragment in sentence.strip('.').strip().split():
        words.extend(re.split(" ", space_separated_fragment))
    return [w for w in words if w]


def parse_paragraph_2(p, target_tokens):
    modified_p = re.sub('#', '.', p)
    ss = re.sub('[^a-zA-Z0-9 \n\.]', '', modified_p).strip('.').split('.')
    ls = list(filter(None, ss))
    total_len = len(ls)
    next_tokens = None
    next_tokens_II = None

    if total_len > 0:
        next_tokens = split_by_whitespace(ls[total_len-1])
    if total_len > 1:
        next_tokens_II = split_by_whitespace(ls[total_len-2])

    if next_tokens:
        lst_next = [get_word(w.lower()) for w in next_tokens]
        next_embs = torch.stack(lst_next)
        next_mean = torch.mean(next_embs, 0)
    else:
        next_mean = torch.FloatTensor(1, GLOVE_DIM).zero_()
    if next_tokens_II:
        lst_next_II = [get_word(w.lower()) for w in next_tokens_II]
        next_embs_II = torch.stack(lst_next_II)
        next_mean_II = torch.mean(next_embs_II, 0)
    else:
        next_mean_II = torch.FloatTensor(1, GLOVE_DIM).zero_()
    return next_mean, next_mean_II


def parse_paragraph_3(p, target_tokens):  # <--- 3
    modified_p = re.sub('#', '.', p)
    ss = re.sub('[^a-zA-Z0-9 \n\.]', '', modified_p).strip('.').split('.')
    # print(ss)
    ls = list(filter(None, ss))
    total_len = len(ls)
    next_tokens = None
    next_tokens_II = None

    if total_len > 0:
        next_tokens = split_by_whitespace(ls[total_len-1])
    if total_len > 1:
        next_tokens_II = split_by_whitespace(ls[total_len-2])
    if total_len > 2:
        next_tokens_III = split_by_whitespace(ls[total_len-3])
    # print(next_tokens, next_tokens_II, next_tokens_III)
    if next_tokens:
        lst_next = [get_word(w.lower()) for w in next_tokens]
        next_embs = torch.stack(lst_next)
        next_mean = torch.mean(next_embs, 0)
    else:
        next_mean = torch.FloatTensor(1, GLOVE_DIM).zero_()
    if next_tokens_II:
        lst_next_II = [get_word(w.lower()) for w in next_tokens_II]
        next_embs_II = torch.stack(lst_next_II)
        next_mean_II = torch.mean(next_embs_II, 0)
    else:
        next_mean_II = torch.FloatTensor(1, GLOVE_DIM).zero_()
    if next_tokens_III:
        lst_next_III = [get_word(w.lower()) for w in next_tokens_III]
        next_embs_III = torch.stack(lst_next_III)
        next_mean_III = torch.mean(next_embs_III, 0)
    else:
        next_mean_III = torch.FloatTensor(1, GLOVE_DIM).zero_()   
    return next_mean, next_mean_II, next_mean_III


def load_dataset(input1, input2):
    input_df1 = pd.read_csv(input1, sep='\t')
    input_df2 = pd.read_csv(input2, sep='\t')
    dict_item_paragraph = input_df2[['Item_ID', '20-b']].groupby('Item_ID')['20-b'].apply(list).to_dict()
    dict_sentence_scores = input_df1[['Item', 'Rating']].groupby('Item')['Rating'].apply(list).to_dict()
    dict_item_mean_score = dict()
    for (k, v) in dict_sentence_scores.items():
        values = np.array(v)
        dict_item_mean_score[k] = np.mean(values)
    dict_item_sentence = input_df1[['Item', 'Sentence']].drop_duplicates().groupby('Item')['Sentence'].apply(list).to_dict()
    return (dict_item_mean_score, dict_item_sentence, dict_item_paragraph)


def single_sentence(contents, contexts):
    f = open('./embs_single.csv', 'w')
    head_line = "Item_ID\tVector_Representation\n"
    f.write(head_line)
    for (k, v) in contents.items():  # <-- only the target
        curr_emb, _ = get_sentence(v[0])
        # content_embs.append(curr_emb)
        curr_emb_list = curr_emb.tolist()
        temp = [format(flt) for flt in curr_emb_list]
        curr_line = k + '\t'
        curr_line += ",".join(temp)
        f.write(curr_line+"\n")
    f.close()


def single_sentence_npy(contents, contexts):
    sentence_embs = []
    i = 0
    for (k, v) in contexts.items():
        # i += 1
        # if (i > 10):
        #     break
        # print(k, v[0])
        curr_emb, _ = get_sentence(v[0])
        curr_emb_list = curr_emb.tolist()
        sentence_embs.append(curr_emb_list)
    sentence_embs_np = np.array(sentence_embs)
    print(sentence_embs_np.shape)
    np.save('./embs_paragraph_rand_unk_2.npy', sentence_embs_np)


def two_precedings(contents, contexts):
    f = open('./embs_two.csv', 'w')
    head_line = "Item_ID\tVector_Representation\n"
    f.write(head_line)
    for (k, v) in contents.items():
        curr_emb, target_tokens = get_sentence(v[0])
        next_emb, next_emb_2 = parse_paragraph_2(contexts[k][0], target_tokens)
        if torch.eq(next_emb, NOT_EXIST).all() and torch.eq(next_emb_2, NOT_EXIST).all():
            content_embs = curr_emb
        elif torch.eq(next_emb_2, NOT_EXIST).all():
            content_embs = curr_emb*np.float64(0.6)+next_emb*np.float64(0.4)
        else:
            content_embs = curr_emb*np.float64(0.6)+next_emb*np.float64(0.2)+next_emb_2*np.float64(0.2)
        curr_emb_list = content_embs.tolist()
        temp = [format(flt) for flt in curr_emb_list]
        curr_line = k + '\t'
        curr_line += ",".join(temp)
        f.write(curr_line+"\n")
    f.close()


def three_precedings(contents, contexts):
    f = open('./embs_three.csv', 'w')
    head_line = "Item_ID\tVector_Representation\n"
    f.write(head_line)
    for (k, v) in contents.items():
        curr_emb, target_tokens = get_sentence(v[0])
        next_emb, next_emb_2, next_emb_3 = parse_paragraph_3(contexts[k][0], target_tokens)
        if torch.eq(next_emb, NOT_EXIST).all() and torch.eq(next_emb_2, NOT_EXIST).all() and torch.eq(next_emb_3, NOT_EXIST).all():
            content_embs = curr_emb
        elif torch.eq(next_emb_2, NOT_EXIST).all() and torch.eq(next_emb_3, NOT_EXIST).all():
            # content_embs.append(curr_emb*np.float64(1.0/2)+next_emb*np.float64(1.0/2))
            content_embs = curr_emb*np.float64(0.6)+next_emb*np.float64(0.4)
        elif torch.eq(next_emb_3, NOT_EXIST).all():
            # content_embs.append(curr_emb*np.float64(1.0/2)+next_emb*np.float64(1.0/2/2)+next_emb_2*np.float64(1.0/2/2))
            content_embs = curr_emb*np.float64(0.6)+next_emb*np.float64(0.2)+next_emb_2*np.float64(0.2)
        else:
            # content_embs.append(curr_emb*np.float64(1.0/2)+next_emb*np.float64(1.0/2/2)+next_emb_2*np.float64(1.0/2/2/2)+next_emb_3*np.float(1.0/2/2/2))
            content_embs = curr_emb*np.float64(0.6)+next_emb*np.float64(0.2)+next_emb_2*np.float64(0.2/2)+next_emb_3*np.float(0.2/2)
        curr_emb_list = content_embs.tolist()
        temp = [format(flt) for flt in curr_emb_list]
        curr_line = k + '\t'
        curr_line += ",".join(temp)
        f.write(curr_line+"\n")
    f.close()


def main():
    parser = argparse.ArgumentParser(
        description='Begin to generate/read word embeddings')
    parser.add_argument('--some_db_dir', dest='some_db_dir', type=str, default='./some_database.csv', help='specify the path to some_database csv file')
    parser.add_argument('--swbd_dir', dest='swbd_dir', type=str, default='./swbdext.csv', help='specify the path to swbdext csv file')
    parser.add_argument('--num_preceding', dest='num_preceding', type=int, default=0, help='specify the number of preceding sentences we want')
    opt = parser.parse_args()
    print(opt)
    _, contents, contexts = load_dataset(opt.some_db_dir, opt.swbd_dir)
    if opt.num_preceding == 0:
        single_sentence_npy(contents, contexts)
    elif opt.num_preceding == 2:
        two_precedings(contents, contexts)
    elif opt.num_preceding == 3:
        three_precedings(contents, contexts)


if __name__ == "__main__":
    main()
