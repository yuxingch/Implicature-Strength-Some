import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import defaultdict
from models import split_by_whitespace, RatingModel, get_sentence
from models import parse_paragraph_2, parse_paragraph_3
import torch
# from vocab import get_glove, build_char_dict

GLOVE_DIM = 100
NOT_EXIST = torch.FloatTensor(1, GLOVE_DIM).zero_()


def load_dataset(some_fulldataset):
    input_df = pd.read_csv(some_fulldataset, sep=',')
    dict_sentence_strength = input_df[['Item', 'StrengthSome']].groupby('Item')['StrengthSome'].apply(list).to_dict()
    dict_item_mean_strength = dict()
    for (k, v) in dict_sentence_strength.items():
        values = np.array(v)
        dict_item_mean_strength[k] = np.mean(values)
    dict_item_sentence = input_df[['Item', 'Sentence']].drop_duplicates().groupby('Item')['Sentence'].apply(list).to_dict()
    return dict_item_mean_strength, dict_item_sentence


def load_paragraph(input_file):
    input_df = pd.read_csv(input_file, sep='\t')
    dict_item_paragraph = input_df[['Item_ID', '20-b']].groupby('Item_ID')['20-b'].apply(list).to_dict()
    return dict_item_paragraph


def main():
    labels, contents = load_dataset('./some_fulldataset.csv')
    contexts = load_paragraph('./swbdext.csv')
    curr_max = 0
    curr_min = 100
    original_labels = []
    for (k, v) in labels.items():
        if v > curr_max:
            curr_max = v
        if v < curr_min:
            curr_min = v
    normalized_labels = []
    max_diff = curr_max - curr_min
    keys = []
    for (k, v) in labels.items():
        keys.append(k)
        original_labels.append(float(v))
        labels[k] = (float(v) - curr_min) / max_diff
        normalized_labels.append(labels[k])

    content_embs = []
    plain_embs = []
    # for (k, v) in contents.items():
    #     curr_emb, _ = get_sentence(v[0])
    #     content_embs.append(curr_emb)
    for (k, v) in contents.items():  # <-- Two Sentences
        curr_emb, target_tokens = get_sentence(v[0])
        next_emb, next_emb_2 = parse_paragraph_2(contexts[k][0], target_tokens)
        if torch.eq(next_emb, NOT_EXIST).all() and torch.eq(next_emb_2, NOT_EXIST).all():
            content_embs.append(curr_emb)
        # elif torch.eq(next_emb, NOT_EXIST).all():
        #     content_embs.append(curr_emb*np.float64(0.6)+next_emb_2*np.float64(0.4))
        elif torch.eq(next_emb_2, NOT_EXIST).all():
            content_embs.append(curr_emb*np.float64(0.6)+next_emb*np.float64(0.4))
        else:
            content_embs.append(curr_emb*np.float64(0.6)+next_emb*np.float64(0.2)+next_emb_2*np.float64(0.2))
        # print(curr_emb, content_embs[len(content_embs)-1])
        plain_embs.append(curr_emb)
    print(curr_min, curr_max)
    # s_model = RatingModel("./strength")
    # s_model.train(torch.stack(content_embs), np.array(normalized_labels))

    s_model_100 = RatingModel("./strength", "./strength/Model_2S/RNet_epoch_100.pth", is_train=False)
    preds_100 = s_model_100.evaluate(torch.stack(content_embs), max_diff, curr_min)
    s_model = RatingModel("./strength", "./strength/Model_2S/RNet_epoch_500.pth", is_train=False)
    preds_500 = s_model.evaluate(torch.stack(content_embs), max_diff, curr_min)
    s_model = RatingModel("./strength", "./strength/Model_2S/RNet_epoch_1000.pth", is_train=False)
    preds_1000 = s_model.evaluate(torch.stack(content_embs), max_diff, curr_min)
    print(np.corrcoef(preds_1000, np.array(original_labels)))
    x_axis = np.arange(len(original_labels))
    plt.scatter(preds_100, np.array(original_labels), s=5, c='r', label="100 epochs")
    plt.scatter(preds_500, np.array(original_labels), s=5, c='b',label="500 epochs")
    plt.scatter(preds_1000, np.array(original_labels), s=5, c='y', label="1000 epochs")
    # print(preds.shape[0], len(original_labels))
    plt.xlim(1, 7)
    plt.ylim(1, 7)
    plt.xlabel('predictions')
    plt.ylabel('real determiner strength')

    function = [i for i in range(1, 1001, 7)]
    plt.plot(function, function, c='k', label="real determiner strength = preds")
    plt.legend()
    plt.title("Target sentence only, determiner strength")
    plt.show()
    return


def load_info(input_file):
    input_df = pd.read_csv(input_file, sep='\t')
    # modification
    dict_item_modified = input_df[['Item_ID', 'redInfoStatus']].groupby('Item_ID')['redInfoStatus'].apply(list).to_dict()
    # new/old/NA


def word_level():
    # partitive/modification/etc.
    return


if __name__ == "__main__":
    main()
    # word_level()
