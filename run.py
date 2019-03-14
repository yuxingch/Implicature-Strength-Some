import argparse
from collections import defaultdict
from datetime import datetime
import os
import pprint
import random
import re

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
from models import parse_paragraph_2, parse_paragraph_3
from utils import mkdir_p


cfg = edict()
# from vocab import get_glove, build_char_dict

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
cfg.ELMO_MODE = 'concat'
cfg.SAVE_PREDS = False
cfg.BATCH_ITEM_NUM = 29

# Training options
cfg.TRAIN = edict()
cfg.TRAIN.FLAG = True
cfg.TRAIN.BATCH_SIZE = 32
cfg.TRAIN.TOTAL_EPOCH = 200
cfg.TRAIN.INTERVAL = 20
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


def load_dataset_plus(input0, input1, input2, t):
    input_df0 = pd.read_csv(input0, sep='\t')
    input_df1 = pd.read_csv(input1, sep=',')
    input_df2 = pd.read_csv(input2, sep='\t')
    dict_item_sentence_raw = input_df0[['Item', 'Sentence']].drop_duplicates().groupby('Item')['Sentence'].apply(list).to_dict()
    dict_item_paragraph_raw = input_df2[['Item_ID', '20-b']].groupby('Item_ID')['20-b'].apply(list).to_dict()
    if t == 'strength':
        dict_item_mean_score_raw = input_df1[['Item', 'StrengthSome']].groupby('Item')['StrengthSome'].apply(list).to_dict()
    else:
        dict_item_mean_score_raw = input_df1[['Item', 'Rating']].groupby('Item')['Rating'].apply(list).to_dict()
    dict_item_modification_raw = input_df1[['Item', 'Modification']].groupby('Item')['Modification'].apply(list).to_dict()
    dict_item_subject_raw = input_df1[['Item', 'Subjecthood']].groupby('Item')['Subjecthood'].apply(list).to_dict()
    dict_item_partitive_raw = input_df1[['Item', 'Partitive']].groupby('Item')['Partitive'].apply(list).to_dict()
    dict_item_mean_score = dict()
    dict_item_sentence = dict()
    dict_item_paragraph = dict()
    dict_item_partitive = dict()
    dict_item_modification = dict()
    dict_item_subject = dict()
    for (k, v) in dict_item_mean_score_raw.items():
        dict_item_mean_score[k] = v[0]
        dict_item_sentence[k] = dict_item_sentence_raw[k]
        dict_item_paragraph[k] = dict_item_paragraph_raw[k]
        dict_item_partitive[k] = dict_item_partitive_raw[k]
        dict_item_modification[k] = dict_item_modification_raw[k]
        dict_item_subject[k] = dict_item_subject_raw[k]
    return (dict_item_mean_score, dict_item_sentence, dict_item_paragraph,
            dict_item_partitive, dict_item_modification, dict_item_subject)


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
    parser.add_argument('--sn', dest='sentence_num', default=0, type=int)
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
        if opt.sentence_num != 0:
            cfg.SINGLE_SENTENCE = False
        cfg.SAVE_PREDS = opt.save_preds
        cfg.EXPERIMENT_NAME = opt.experiment_name
        if not opt.mode == 'train':
            cfg.TRAIN.FLAG = False
            cfg.MODE = opt.mode

    print('Using configurations:')
    pprint.pprint(cfg)

    curr_path = "./datasets/seed_" + str(cfg.SEED)
    is_train = True
    if cfg.EXPERIMENT_NAME == "":
        cfg.EXPERIMENT_NAME = datetime.now().strftime('%m_%d_%H_%M')
    if opt.mode == 'analyze':
        # pass
        eval_path = "./" + cfg.EXPERIMENT_NAME + "_" + cfg.PREDICTION_TYPE + "_" + str(cfg.SEED)
        if cfg.IS_RANDOM:
            eval_path += "_random"
            load_path = eval_path + "/Model_" + str(opt.sentence_num) + "S"
        else:
            load_path = eval_path + "/Model_" + str(opt.sentence_num) + "S"
        epoch_to_analyze = [0, 1, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
        epoch_npy = []
        for epoch in epoch_to_analyze:
            cfg.RESUME_DIR = load_path + "/RNet_epoch_" + str(epoch) + ".pth"
            r_model_analyze = RatingModel(cfg, eval_path)
            epoch_npy.append(r_model_analyze.analyze())
        # -- save numpy file
        # np.save('plus', epoch_npy[3][:, :100])
        # return

        # fig = plt.figure(figsize=(64, GLOVE_DIM))
        # fig = plt.figure(figsize=(1, 32))
        num_cols = 2
        num_kernels = epoch_npy[4].shape[0]
        num_rows = 1 + num_kernels // num_cols
        fig = plt.figure(figsize=(num_cols,num_rows))
        for i in range(epoch_npy[4].shape[0]):
            ax1 = fig.add_subplot(num_rows,num_cols,i+1)
            tensor = np.swapaxes(epoch_npy[4][i],0,1)
            ax1.imshow(tensor)
            ax1.axis('off')
            ax1.set_title(i)
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.show()
        # fig = plt.figure(figsize=(64, GLOVE_DIM+3))
        # columns = 4
        # rows = int(len(epoch_to_analyze) / columns)
        # ax = []
        # for i in range(1, columns*rows+1):
            # img = epoch_npy[i-1]
            # column_sum = np.sum(img, axis=0)
            # print(column_sum.argsort()[-4:][::-1])#, column_sum/64)  #[-4:][::-1])
            # column_max, column_min = np.argmax(column_sum), np.argmin(column_sum)
            # ax.append(fig.add_subplot(rows, columns, i))
            # ax[-1].set_title(opt.t + "_UNK_" + str(GLOVE_DIM) + "_epoch" + str(epoch_to_analyze[i-1]) + "_score")
            # ax[-1].axvline(x=column_max, color='maroon')
            # ax[-1].axvline(x=column_min, color='lightsteelblue')
            # plt.imshow(img)
            # print(i, np.mean(img))

        # -- plot the overall heatmap
        # heat_map = np.zeros_like(epoch_npy[0])
        # for i in range(1, columns*rows):
        #     heat_map += np.absolute(epoch_npy[i] - epoch_npy[i-1])
        # heat_min, heat_max = np.amin(heat_map), np.max(heat_map)
        # heat_map = (heat_map - heat_min) / (heat_max - heat_min) * 255.0
        # plt.imshow(heat_map, cmap='RdPu')
        # plt.title('most frequently updated weights (scaled from 0 to 255)')
        # plt.colorbar()

        # -- difference between epoch 0 and epoch 60
        # w = epoch_npy[3] - epoch_npy[0]
        # w_max, w_min = np.amax(w), np.amin(w)
        # w_normalized = (w - w_min) / (w_max - w_min) * 255.0
        plt.show()
        return
    elif cfg.MODE == 'train':
        load_db = curr_path + "/train_db.csv"
    elif cfg.MODE == 'eval':
        load_db = curr_path + "/eval_db.csv"
        is_train = False
    elif cfg.MODE == 'all':
        is_train = False
        load_db = curr_path + "/all_db.csv"
    # labels, contents, contexts, part, mod, sub = load_dataset("./some_database.csv", load_db, "./swbdext.csv", opt.t)
    labels, contents, contexts = load_dataset(cfg.SOME_DATABASE, load_db,
                                              "./swbdext.csv", cfg.PREDICTION_TYPE)

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
    cfg.BATCH_ITEM_NUM = len(normalized_labels)//cfg.TRAIN.BATCH_SIZE

    content_embs = []
    plain_embs = []
    content_embs = []
    content_embs_np = None
    content_embs_stack = None

    if opt.sentence_num == 0:
        NUMPY_PATH = './datasets/seed_' + str(cfg.SEED)
        if cfg.IS_ELMO:
            NUMPY_PATH += '/elmo_' + cfg.ELMO_MODE + '/embs_' + cfg.MODE + '.npy'
        else:
            NUMPY_PATH += '/embs_' + cfg.MODE + '.npy'
        if os.path.isfile(NUMPY_PATH):
            content_embs_np = np.load(NUMPY_PATH)
            content_embs_stack = torch.from_numpy(content_embs_np)
        else:
            for (k, v) in tqdm(contents.items(), total=len(contents)):  # <-- only the target
                # curr_emb, _ = get_sentence_2d(v[0])
                # context_v = contexts[k]
                if cfg.IS_ELMO:
                    # elmo
                    curr_emb, _ = get_sentence_elmo(v[0], embedder=embedder, elmo_mode=cfg.ELMO_MODE)
                else:
                    # curr_emb, _ = get_sentence(context_v[0])
                    curr_emb, _ = get_sentence(v[0])

                # curr_plus = torch.tensor([float(part[k][0]), float(mod[k][0]), float(sub[k][0])])
                # print(k, torch.cat((curr_emb, curr_plus)))
                # return
                content_embs.append(curr_emb)
                # content_embs.append(torch.cat((curr_emb, curr_plus)))
            content_embs_stack = torch.stack(content_embs)
            np.save(NUMPY_PATH, content_embs_stack.numpy())
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

    fake_embs = None
    if opt.random_vector:
        print("randomized word vectors")
        if is_train:
            fake_embs = random_input(954)
        else:
            fake_embs = random_input(408)

    if cfg.TRAIN.FLAG:
        save_path = "./" + cfg.EXPERIMENT_NAME + "_" + cfg.PREDICTION_TYPE + "_" + str(cfg.SEED)
        if cfg.IS_RANDOM:
            save_path += "_random"
            print(save_path)
            r_model = RatingModel(cfg, save_path, sn=opt.sentence_num)
            r_model.train(fake_embs, np.array(normalized_labels))
        else:
            print(save_path)
            r_model = RatingModel(cfg, save_path, sn=opt.sentence_num)
            # r_model.train(torch.stack(content_embs), np.array(normalized_labels), prev_epoch=0)
            r_model.train(content_embs_stack, np.array(normalized_labels))
    else:
        eval_path = "./" + cfg.EXPERIMENT_NAME + "_" + cfg.PREDICTION_TYPE + "_" + str(cfg.SEED)
        epoch_lst = [0, 1]
        i = 0
        while i < cfg.TRAIN.TOTAL_EPOCH - cfg.TRAIN.INTERVAL + 1:
            i += cfg.TRAIN.INTERVAL
            epoch_lst.append(i)
        if cfg.IS_RANDOM:
            eval_path += "_random"
            load_path = eval_path + "/Model_" + str(opt.sentence_num) + "S"
            for epoch in epoch_lst:
                cfg.RESUME_DIR = load_path + "/RNet_epoch_" + format(epoch)+".pth"
                r_model_decay = RatingModel(cfg, eval_path)
                preds_decay = r_model_decay.evaluate(fake_embs, max_diff, curr_min)
        else:
            load_path = eval_path + "/Model_" + str(opt.sentence_num) + "S"
            max_epoch_dir = None
            max_value = -1.0
            max_epoch = None
            for epoch in epoch_lst:
                cfg.RESUME_DIR = load_path + "/RNet_epoch_" + format(epoch)+".pth"
                r_model_decay = RatingModel(cfg, eval_path)
                # preds_decay = r_model_decay.evaluate(torch.stack(content_embs), max_diff, curr_min)
                preds_decay = r_model_decay.evaluate(content_embs_stack, max_diff, curr_min)
                curr_coeff = np.corrcoef(preds_decay, np.array(original_labels))[0, 1]
                print(curr_coeff)
                if max_value < curr_coeff:
                    max_value = curr_coeff
                    max_epoch_dir = cfg.RESUME_DIR
                    max_epoch = epoch
            if cfg.MODE == 'all':
                # save all predictions
                cfg.RESUME_DIR = max_epoch_dir
                r_model_decay = RatingModel(cfg, eval_path)
                preds_decay = r_model_decay.evaluate(content_embs_stack, max_diff, curr_min)
                # print('Current max: ', preds_decay.shape, max_epoch, max_epoch_dir, max_value)

                f = open('./0313_rating/all_preds_'+format(max_epoch)+'.csv', 'w')
                head_line = "Item_ID\toriginal_mean\tpredicted\n"
                f.write(head_line)
                for i in range(len(keys)):
                    k = keys[i]
                    ori = original_labels[i]
                    pre = preds_decay[i]
                    curr_line = k + '\t' + format(ori) + '\t' + format(pre)
                    f.write(curr_line+"\n")
                f.close()

        # save hidden vector
        # f = open('./0216_rating/train_first_hidden_layer_epoch80.csv', 'w')
        # head_line = "Item_ID\tHidden_Vector_Representation\n"
        # f.write(head_line)
        # counter = 0
        # for (k, v) in contents.items():  # <-- only the target
        #     curr_hidden = all_hiddens[counter, :]
        #     curr_hidden_list = curr_hidden.tolist()
        #     temp = [format(flt) for flt in curr_hidden_list]
        #     curr_line = k + '\t'
        #     curr_line += ",".join(temp)
        #     f.write(curr_line+"\n")
        #     counter += 1
        # f.close()

        # correlation coefficient
        # flops = len(original_labels)//32
        # part_labels = original_labels[:32*flops]

        # print(np.corrcoef(preds_decay_0, np.array(part_labels)))
        # print(np.corrcoef(preds_decay_1, np.array(part_labels)))
        # print(np.corrcoef(preds_decay, np.array(part_labels)))

        # print(np.corrcoef(preds_decay_0, np.array(original_labels)))
        # print(np.corrcoef(preds_decay_1, np.array(original_labels)))
        # print(np.corrcoef(preds_decay, np.array(original_labels)))

        # save predictions as .csv file if needed
        # if opt.save_preds == 1:
        #     new_file_path = './0216_' + opt.t
        #     mkdir_p(new_file_path)
        #     new_file_name = new_file_path + '/train_preds_rating_seed' + str(opt.seed_nm) + '_' + str(GLOVE_DIM) + 'd_' + str(opt.sentence_num) + 'sn_epoch80.csv'
        #     f = open(new_file_name, 'w')
        #     head_line = "Item_ID\toriginal_mean\tpredicted\n"
        #     print(f'Start writing predictions to file:\n{new_file_name}\n...')
        #     f.write(head_line)
        #     for i in range(len(keys)):
        #     # for i in range(flops*32):
        #         k = keys[i]
        #         ori = original_labels[i]
        #         pre = preds_decay[i]
        #         curr_line = k + '\t' + format(ori) + '\t' + format(pre)
        #         f.write(curr_line+"\n")
        #     f.close()

        # plot
        # plt.scatter(preds_decay_0, np.array(part_labels), s=5, c='y', label="initial")
        # plt.scatter(preds_decay_1, np.array(part_labels), s=5, c='r', label="1 epoch")
        # plt.scatter(preds_decay, np.array(part_labels), s=5, c='b', label="160 epochs")
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
