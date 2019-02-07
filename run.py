import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import defaultdict
from models import split_by_whitespace, RatingModel, get_sentence
from models import parse_paragraph_2, parse_paragraph_3
import torch
import random
import argparse
from utils import mkdir_p
# from vocab import get_glove, build_char_dict

GLOVE_DIM = 100
NOT_EXIST = torch.FloatTensor(1, GLOVE_DIM).zero_()


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
    return (dict_item_mean_score, dict_item_sentence, dict_item_paragraph)


def random_input(num_examples):
    res = []
    for i in range(num_examples):
        lst = []
        for j in range(GLOVE_DIM):
            lst.append(round(random.uniform(-1, 1), 16))
        res.append(lst)
    return torch.Tensor(res)


def main():
    parser = argparse.ArgumentParser(
        description='Run ...')
    parser.add_argument('--seed', dest='seed_nm', default=0, type=int)
    parser.add_argument('--mode', dest='mode', default='train')
    parser.add_argument('--t', dest='t', default='rating')
    parser.add_argument('--random', dest='random_vector', default=False)
    parser.add_argument('--sn', dest='sentence_num', default=0, type=int)
    parser.add_argument('--save_preds', dest='save_preds', default=1)
    opt = parser.parse_args()
    print(opt)
    some_database = "./some_database.csv"
    curr_path = "./datasets/seed_" + str(opt.seed_nm)
    is_train = True
    if opt.mode == 'analyze':
        # pass
        eval_path = "./emb" + str(GLOVE_DIM) + "_" + opt.t + "_" + str(opt.seed_nm)
        r_model_analyze = None
        if opt.random_vector:
            eval_path += "_random"
            load_path = eval_path + "/Model_" + str(opt.sentence_num) + "S"
        else:
            load_path = eval_path + "/Model_" + str(opt.sentence_num) + "S"
        epoch_to_analyze = [0, 20, 40, 60, 80, 100]
        epoch_npy = []
        for epoch in epoch_to_analyze:
            r_model_analyze = RatingModel(eval_path, load_checkpoint=load_path + "/RNet_epoch_" + str(epoch) + ".pth", is_train=False)
            epoch_npy.append(r_model_analyze.analyze())
        # fig = plt.figure(figsize=(64, GLOVE_DIM))
        # fig = plt.figure(figsize=(32, 64))
        fig = plt.figure(figsize=(1, 32))
        columns = 2
        rows = int(len(epoch_to_analyze) / columns)
        ax = []
        for i in range(1, columns*rows+1):
            img = epoch_npy[i-1]
            column_sum = np.sum(img, axis=0)
            # print(column_sum.argsort())#[-4:][::-1])
            column_max, column_min = np.argmax(column_sum), np.argmin(column_sum)
            ax.append(fig.add_subplot(rows, columns, i))
            ax[-1].set_title(opt.t + "_" + str(GLOVE_DIM) + "_epoch" + str(epoch_to_analyze[i-1]) + "_score")
            ax[-1].axvline(x=column_max, color='maroon')
            ax[-1].axvline(x=column_min, color='lightsteelblue')
            plt.imshow(img)
            print(i, np.mean(img))
        plt.show()
        return
    elif opt.mode == 'train':
        load_db = curr_path + "/train_db.csv"
    elif opt.mode == 'eval':
        load_db = curr_path + "/eval_db.csv"
        is_train = False
    elif opt.mode == 'all':
        is_train = False
        load_db = curr_path + "/all_db.csv"
    labels, contents, contexts = load_dataset("./some_database.csv", load_db, "./swbdext.csv", opt.t)

    curr_max = 0
    curr_min = 100
    original_labels = []
    for (k, v) in labels.items():
        if v > curr_max:
            curr_max = v
        if v < curr_min:
            curr_min = v
    # normalize the values
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
    content_embs = []

    if opt.sentence_num == 0:
        for (k, v) in contents.items():  # <-- only the target
            curr_emb, _ = get_sentence(v[0])
            content_embs.append(curr_emb)
    elif opt.sentence_num == 2:
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
    elif opt.sentence_num == 3:
        for (k, v) in contents.items():  # <-- Three Sentences
            keys.append(k)
            curr_emb, target_tokens = get_sentence(v[0])
            next_emb, next_emb_2, next_emb_3 = parse_paragraph_3(contexts[k][0], target_tokens)
            if torch.eq(next_emb, NOT_EXIST).all() and torch.eq(next_emb_2, NOT_EXIST).all() and torch.eq(next_emb_3, NOT_EXIST).all():
                content_embs.append(curr_emb)
            elif torch.eq(next_emb_2, NOT_EXIST).all() and torch.eq(next_emb_3, NOT_EXIST).all():
                # content_embs.append(curr_emb*np.float64(1.0/2)+next_emb*np.float64(1.0/2))
                content_embs.append(curr_emb*np.float64(0.6)+next_emb*np.float64(0.4))
            elif torch.eq(next_emb_3, NOT_EXIST).all():
                # content_embs.append(curr_emb*np.float64(1.0/2)+next_emb*np.float64(1.0/2/2)+next_emb_2*np.float64(1.0/2/2))
                content_embs.append(curr_emb*np.float64(0.6)+next_emb*np.float64(0.2)+next_emb_2*np.float64(0.2))
            else:
                # content_embs.append(curr_emb*np.float64(1.0/2)+next_emb*np.float64(1.0/2/2)+next_emb_2*np.float64(1.0/2/2/2)+next_emb_3*np.float(1.0/2/2/2))
                content_embs.append(curr_emb*np.float64(0.6)+next_emb*np.float64(0.2)+next_emb_2*np.float64(0.2/2)+next_emb_3*np.float(0.2/2))
    else:
        print("sentence_num is not valid.")
        return

    if opt.random_vector:
        print("randomized word vectors")
        if is_train:
            fake_embs = random_input(954)
        else:
            fake_embs = random_input(408)

    if is_train:
        save_path = "./emb" + str(GLOVE_DIM) + "_" + opt.t + "_" + str(opt.seed_nm)
        if opt.random_vector:
            save_path += "_random"
            print(save_path)
            r_model = RatingModel(save_path, sn=opt.sentence_num)
            r_model.train(fake_embs, np.array(normalized_labels))
        else:
            print(save_path)
            r_model = RatingModel(save_path, sn=opt.sentence_num)
            r_model.train(torch.stack(content_embs), np.array(normalized_labels))
    else:
        eval_path = "./emb" + str(GLOVE_DIM) + "_" + opt.t + "_" + str(opt.seed_nm)
        if opt.random_vector:
            eval_path += "_random"
            load_path = eval_path + "/Model_" + str(opt.sentence_num) + "S"
            r_model_decay = RatingModel(eval_path, load_checkpoint=load_path + "/RNet_epoch_60.pth", is_train=False)
            preds_decay = r_model_decay.evaluate(fake_embs, max_diff, curr_min)
            r_model_decay_1 = RatingModel(eval_path, load_checkpoint=load_path + "/RNet_epoch_1.pth", is_train=False)
            preds_decay_1 = r_model_decay_1.evaluate(fake_embs, max_diff, curr_min)
            r_model_decay_0 = RatingModel(eval_path, load_checkpoint=load_path + "/RNet_epoch_0.pth", is_train=False)
            preds_decay_0 = r_model_decay_0.evaluate(fake_embs, max_diff, curr_min)
        else:
            load_path = eval_path + "/Model_" + str(opt.sentence_num) + "S"
            r_model_decay = RatingModel(eval_path, load_checkpoint=load_path + "/RNet_epoch_120.pth", is_train=False)
            preds_decay = r_model_decay.evaluate(torch.stack(content_embs), max_diff, curr_min)
            r_model_decay_1 = RatingModel(eval_path, load_checkpoint=load_path + "/RNet_epoch_1.pth", is_train=False)
            preds_decay_1 = r_model_decay_1.evaluate(torch.stack(content_embs), max_diff, curr_min)
            r_model_decay_0 = RatingModel(eval_path, load_checkpoint=load_path + "/RNet_epoch_0.pth", is_train=False)
            preds_decay_0 = r_model_decay_0.evaluate(torch.stack(content_embs), max_diff, curr_min)

        # correlation coefficient
        print(np.corrcoef(preds_decay_0, np.array(original_labels)))
        print(np.corrcoef(preds_decay_1, np.array(original_labels)))
        print(np.corrcoef(preds_decay, np.array(original_labels)))

        # save predictions as .csv file if needed
        if opt.save_preds == 1:
            new_file_path = './preds_' + opt.t
            mkdir_p(new_file_path)
            new_file_name = new_file_path + '/preds_rating_seed' + str(opt.seed_nm) + '_' + str(GLOVE_DIM) + 'd_' + str(opt.sentence_num) + 'sn_epoch80.csv'
            f = open(new_file_name, 'w')
            head_line = "Item_ID\toriginal_mean\tpredicted\n"
            print(f'Start writing predictions to file:\n{new_file_name}\n...')
            f.write(head_line)
            for i in range(len(keys)):
                k = keys[i]
                ori = original_labels[i]
                pre = preds_decay[i]
                curr_line = k + '\t' + format(ori) + '\t' + format(pre)
                f.write(curr_line+"\n")
            f.close()

        # plot
        # plt.scatter(preds_decay_0, np.array(original_labels), s=5, c='y', label="initial")
        # plt.scatter(preds_decay_1, np.array(original_labels), s=5, c='r', label="1 epoch")
        # plt.scatter(preds_decay, np.array(original_labels), s=5, c='b', label="60 epochs")
        # plt.xlim(1, 7)
        # plt.ylim(1, 7)
        # plt.xlabel('predictions')
        # plt.ylabel('real ratings/strengths')
        # function = [i for i in range(1, 1001, 7)]
        # plt.plot(function, function, c='k', label="real rating = preds")
        # plt.legend()
        # plt.title("Target sentence only, implicature strength rating")
        # plt.show()
    return

if __name__ == "__main__":
    main()
