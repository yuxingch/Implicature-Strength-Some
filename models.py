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

import tensorflow as tf

from utils import mkdir_p, weights_init, save_model
ssl._create_default_https_context = ssl._create_unverified_context

GLOVE_DIM = 100

glove = vocab.GloVe(name='6B', dim=GLOVE_DIM)
# print('Loaded {} words'.format(len(glove.itos)))

torch.manual_seed(7)

_UNK = torch.randn(GLOVE_DIM,)
_PAD = torch.randn(GLOVE_DIM,)


def build_state_dict(config_net):
    """Build dictionary to store the state of our net"""
    return torch.load(config_net, map_location=lambda storage, loc: storage)


def write_summary(value, tag, summary_writer, global_step):
    """Write a single summary value to tensorboard"""
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(summary, global_step)


class RatingModel(object):

    def __init__(self, output_dir, load_checkpoint="", is_train=True):
        if is_train:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.log_dir = os.path.join(output_dir, 'Log')
            mkdir_p(self.model_dir)
            mkdir_p(self.log_dir)
            self.summary_writer = tf.summary.FileWriter(self.log_dir)
        
        self.batch_size = 32
        self.total_epoch = 100
        self.load_checkpoint = load_checkpoint
        self.lr = 0.01
    
    def load_network(self):
        from net import RateNet
        print('initializing neural net')
        RNet = RateNet(100)
        RNet.apply(weights_init)

        if self.load_checkpoint != "":
            RNet.load_state_dict(build_state_dict(self.load_checkpoint))
            print(f'Load from: {self.load_checkpoint}')
        
        return RNet

    def train(self, word_embs, labels):
        labels = np.expand_dims(labels, axis=1) 
        RNet = self.load_network()
        optimizer = optim.Adam(RNet.parameters(), lr=self.lr, betas=(0.9, 0.999))
        # print(word_embs.size())
        count = 0
        epoch = 0
        while (epoch < self.total_epoch):
            epoch += 1
            start_t = time.time()
            batch_inds = list(BatchSampler(SequentialSampler(word_embs),
                                           batch_size=self.batch_size,
                                           drop_last=True))
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
            if epoch % 10 == 0:
                save_model(RNet, epoch, self.model_dir)
        save_model(RNet, self.total_epoch, self.model_dir)


def get_word(w):
    try:
        result = glove.vectors[glove.stoi[w]]
    except KeyError:
        result = _UNK
    return result

def split_by_whitespace(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(" ", space_separated_fragment))
    return [w for w in words if w]

def get_sentence(s, max_len=40):
    s = re.sub('[^a-zA-Z0-9 \n\.]', '', s)
    raw_tokens = split_by_whitespace(s)
    # print(raw_tokens)
    n = len(raw_tokens)
    if (n < max_len):
        lst = [get_word(w.lower()) for w in raw_tokens]
        # lst += [_PAD] * (max_len - n)
    else:
        raw_tokens = raw_tokens[:max_len]
        lst = [get_word(w.lower()) for w in raw_tokens]
    all_embs = torch.stack(lst)
    return torch.mean(all_embs, 0) # embedding_size


def main():
    result = get_sentence("Good Morning!")
    print(result)

if __name__ == "__main__":
    main()