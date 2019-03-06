from itertools import combinations
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchtext.vocab as vocab
from torch.utils.data.sampler import SequentialSampler, BatchSampler
import ssl
import re
import os
import time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import visdom

from utils import mkdir_p, weights_init, save_model
ssl._create_default_https_context = ssl._create_unverified_context

GLOVE_DIM = 100

glove = vocab.GloVe(name='6B', dim=GLOVE_DIM)
# print('Loaded {} words'.format(len(glove.itos)))

torch.manual_seed(1)

_UNK = torch.randn(GLOVE_DIM,)
_PAD = torch.randn(GLOVE_DIM,)


def build_state_dict(config_net):
    """Build dictionary to store the state of our net"""
    return torch.load(config_net, map_location=lambda storage, loc: storage)['state_dict']


def write_summary(value, tag, summary_writer, global_step):
    """Write a single summary value to tensorboard"""
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(summary, global_step)


class RatingModel(object):

    def __init__(self, output_dir, sn=0, load_checkpoint="", is_train=True):
        if is_train:
            self.model_dir = os.path.join(output_dir, 'Model_' + str(sn) + 'S')
            self.log_dir = os.path.join(output_dir, 'Log_' + str(sn) + 'S')
            mkdir_p(self.model_dir)
            mkdir_p(self.log_dir)
            self.summary_writer = tf.summary.FileWriter(self.log_dir)

        self.batch_size = 32
        self.total_epoch = 200
        self.load_checkpoint = load_checkpoint
        # self.lr = 0.02
        self.lr = 0.02
        self.lr_decay_per_epoch = 20

    def load_network(self):
        from net import RateNet, RateNet2D
        print('initializing neural net')
        # RNet = RateNet(GLOVE_DIM)
        RNet = RateNet2D(GLOVE_DIM)
        RNet.apply(weights_init)

        if self.load_checkpoint != "":
            RNet.load_state_dict(build_state_dict(self.load_checkpoint))
            print(f'Load from: {self.load_checkpoint}')

        return RNet

    def train(self, word_embs, labels, prev_epoch=0):
        labels = np.expand_dims(labels, axis=1) 
        RNet = self.load_network()
        lr = self.lr
        optimizer = optim.Adam(RNet.parameters(), lr=lr, betas=(0.9, 0.999))
        # print(word_embs.size())
        count = 0
        epoch = prev_epoch
        if epoch == 0:
            save_model(RNet, epoch, self.model_dir)
        while (epoch < self.total_epoch):
            epoch += 1
            start_t = time.time()
            batch_inds = list(BatchSampler(SequentialSampler(word_embs),
                                           batch_size=self.batch_size,
                                           drop_last=True))

            if epoch % self.lr_decay_per_epoch == 0:
                # update learning rate
                lr *= 0.8
                print(f'learning rate updated: {lr}')
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            for i, inds in enumerate(batch_inds, 0):
                real_labels = labels[inds]
                curr_batch = word_embs[inds]

                curr_batch_tensor = Variable(curr_batch, requires_grad=True)
                real_label_tensor = Variable(torch.from_numpy(real_labels))

                output_scores = RNet(curr_batch_tensor)

                RNet.zero_grad()
                loss_func = nn.MSELoss()
                loss = loss_func(output_scores, real_label_tensor.type(torch.FloatTensor))
                loss.backward()

                optimizer.step()

                count += 1
                if i % 10 == 0:
                    write_summary(loss, 'loss', self.summary_writer, count)

            end_t = time.time()
            print(f'[{epoch}/{self.total_epoch}][{i}/{len(batch_inds)-1}] Loss: {loss:.4f}'
                    f' Total Time: {(end_t-start_t):.2f}sec')
            if epoch % 20 == 0 or epoch == 1:
                save_model(RNet, epoch, self.model_dir)
        save_model(RNet, self.total_epoch, self.model_dir)

    def evaluate(self, word_embs, max_diff, min_value):
        RNet = self.load_network()
        RNet.eval()

        num_items = word_embs.shape[0]
        # print(num_items)
        batch_size = min(num_items, self.batch_size)

        rating_lst = []
        count = 0
        all_hiddens_list = []
        while count < num_items:
            iend = count + batch_size
            if iend > num_items:
                iend = num_items
                # break
                # count = num_items - batch_size
            curr_batch = Variable(word_embs[count:iend])
            # output_scores, h = RNet(curr_batch)
            output_scores = RNet(curr_batch)
            # all_hiddens_list.append(h)
            # print(output_scores.size())
            for curr_score in output_scores.data.tolist():
                rating_lst.append(curr_score[0]*max_diff+min_value)
            count += batch_size
        # all_hiddens = torch.cat(tuple(all_hiddens_list)).data.numpy()
        return np.array(rating_lst)  #, all_hiddens

    def analyze(self):
        RNet = self.load_network()
        RNet.eval()

        w = RNet.conv1.weight.data.numpy()
        # w_max, w_min = np.amax(w), np.amin(w)
        # w_normalized = (w - w_min) / (w_max - w_min) * 255.0
        # return w_normalized
        return w


def get_word(w):
    try:
        result = glove.vectors[glove.stoi[w]]
    except KeyError:
        result = _UNK
    return result


def split_by_whitespace(sentence):
    words = []
    for space_separated_fragment in sentence.strip('.').strip().split():
        words.extend(re.split(" ", space_separated_fragment))
    return [w for w in words if w]


def get_sentence(s, max_len=40):
    # s = re.sub('[^a-zA-Z0-9 \n\.]', '', s)
    # raw_tokens = split_by_whitespace(s)
    # # print(raw_tokens)
    # n = len(raw_tokens)
    # if (n < max_len):
    #     lst = [get_word(w.lower()) for w in raw_tokens]
    #     # lst += [_PAD] * (max_len - n)
    # else:
    #     raw_tokens = raw_tokens[:max_len]
    #     lst = [get_word(w.lower()) for w in raw_tokens]
    s = s.replace('\'ve', ' \'ve')
    s = s.replace('\'re', ' \'re')
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
        s = s.replace(' oclock ', ' o\'clock ')
        s = s.replace(' ve ', ' \'ve ')
        s = s.replace(' re ', ' \'re ')
        s = s.replace(' ll ', ' \'ll ')
        s = s.replace(' nt ', ' n\'t ')
        s = s.replace(' d ', ' \'d ')
        s = s.replace(' s ', ' \'s ')
        s = s.replace('doeuvres', 'd\'oeuvres')
        s = s.replace('mumblex', 'mumble')
        raw_tokens += split_by_whitespace(s)
    # lst = [get_word(w.lower()) for w in raw_tokens]
    lst = []
    for w in raw_tokens:
        curr_emb = get_word(w.lower())
        if torch.all(torch.eq(curr_emb, _UNK)):
            print(w)
            continue
        else:
            lst.append(curr_emb)
    all_embs = torch.stack(lst)
    return torch.mean(all_embs, 0), raw_tokens  # embedding_size


def get_sentence_2d(s, max_len=32):
    # s = re.sub('[^a-zA-Z0-9 \n\.]', '', s)
    # raw_tokens = split_by_whitespace(s)
    # n = len(raw_tokens)
    # if (n < max_len):
    #     lst = [get_word(w.lower()) for w in raw_tokens]
    #     lst += [_PAD] * (max_len - n)
    # else:
    #     raw_tokens = raw_tokens[:max_len]
    #     lst = [get_word(w.lower()) for w in raw_tokens]
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
    n = len(raw_tokens)
    if (n < max_len):
        lst = [get_word(w.lower()) for w in raw_tokens]
        lst += [_PAD] * (max_len - n)
    else:
        raw_tokens = raw_tokens[:max_len]
        lst = [get_word(w.lower()) for w in raw_tokens]
    all_embs = torch.stack(lst).permute(1, 0)
    return all_embs, raw_tokens


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

# def parse_paragraph(p, target_tokens):
#     ss = re.sub('[^a-zA-Z0-9 \n\.]', '', p).strip('.').split('.')
#     ls = list(filter(None, ss))
#     total_len = len(ls)
#     prev_tokens = None
#     next_tokens = None
#     ptr = 0
#     found = False
#     for s in ls:
#         curr_tokens = split_by_whitespace(s)
#         print(curr_tokens, target_tokens)
#         if curr_tokens == target_tokens:
#             print("hi", ptr)
#             if (ptr+1) < total_len:
#                 next_tokens = split_by_whitespace(ls[ptr+1])
#                 found = True
#             break
#         prev_tokens = curr_tokens
#         ptr += 1

#     if prev_tokens and found:
#         lst_prev = [get_word(w.lower()) for w in prev_tokens]
#         prev_embs = torch.stack(lst_prev)
#         prev_mean = torch.mean(prev_embs, 0)
#     else:
#         prev_mean = torch.IntTensor(1, GLOVE_DIM).zero_()
#     if next_tokens:
#         lst_next = [get_word(w.lower()) for w in next_tokens]
#         next_embs = torch.stack(lst_next)
#         next_mean = torch.mean(next_embs, 0)
#     else:
#         next_mean = torch.IntTensor(1, GLOVE_DIM).zero_()
#     return prev_mean, next_mean


def main():
    result = get_sentence("Good Morning!")
    print(result)

if __name__ == "__main__":
    main()
