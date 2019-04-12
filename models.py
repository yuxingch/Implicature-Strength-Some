from itertools import combinations
import os
import re
import ssl
import time

# ELMo
from allennlp.commands.elmo import ElmoEmbedder
# from allennlp.modules.elmo import Elmo, batch_to_ids
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import tensorflow as tf
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchtext.vocab as vocab
from torch.utils.data.sampler import SequentialSampler, BatchSampler
from torch.nn.utils import clip_grad_value_
from torch.nn.utils.rnn import pack_padded_sequence

from utils import mkdir_p, weights_init, save_model
ssl._create_default_https_context = ssl._create_unverified_context


OPTION_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/" \
              "2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
WEIGHT_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/" \
              "2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

GLOVE_DIM = 100
glove = vocab.GloVe(name='6B', dim=GLOVE_DIM)

IMG_DIR = "/Users/yuxing/Desktop/Stanford/Academic/2018-2019/Spring2019/temp/"

torch.manual_seed(1)

_UNK = torch.randn(GLOVE_DIM,)
_PAD = torch.randn(GLOVE_DIM,)


def build_state_dict(config_net):
    """Build dictionary to store the state of our neural net"""
    return torch.load(config_net, map_location=lambda storage, loc: storage)['state_dict']


def write_summary(value, tag, summary_writer, global_step):
    """Write a single summary value to tensorboard"""
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(summary, global_step)


class RatingModel(object):

    def __init__(self, cfg, output_dir, sn=0):
        """Intialize RatingModel

        Positional arguments:
        cfg -- configuration dictionary
        output_dir -- path to save checkpoints and logs

        Keyword argument:
        sn -- number of sentences taken into consideration (default 0)
        """
        self.cfg = cfg
        if self.cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model_' + str(sn) + 'S')
            self.log_dir = os.path.join(output_dir, 'Log_' + str(sn) + 'S')
            mkdir_p(self.model_dir)
            mkdir_p(self.log_dir)
            self.summary_writer = tf.summary.FileWriter(self.log_dir)

        self.batch_size = self.cfg.TRAIN.BATCH_SIZE
        self.total_epoch = self.cfg.TRAIN.TOTAL_EPOCH
        self.load_checkpoint = self.cfg.RESUME_DIR
        self.lr = self.cfg.TRAIN.LR
        self.lr_decay_per_epoch = self.cfg.TRAIN.LR_DECAY_EPOCH
        self.dropout = [self.cfg.TRAIN.DROPOUT.FC_1, self.cfg.TRAIN.DROPOUT.FC_2]
        self.drop_prob = self.cfg.LSTM.DROP_PROB

    def load_network(self):
        """Initialize the network or load from checkpoint"""
        from net import RateNet, RateNet2D, RateNetELMo, BiLSTMELMo
        print('initializing neural net')
        self.RNet = None
        if self.cfg.IS_ELMO:
            if self.cfg.ELMO_MODE == 'concat':
                elmo_dim = 3072
            else:
                elmo_dim = 1024
            if self.cfg.LSTM.FLAG:
                # elmo_dim, seq_len, hidden_dim, num_layers, drop_prob, dropout
                self.RNet = BiLSTMELMo(elmo_dim, self.cfg.LSTM.SEQ_LEN,
                                       self.cfg.LSTM.HIDDEN_DIM, self.cfg.LSTM.LAYERS,
                                       self.drop_prob, self.dropout)
            else:
                self.RNet = RateNetELMo(elmo_dim, self.dropout)
        else:
            self.RNet = RateNet(self.cfg.GLOVE_DIM, self.dropout)
        self.RNet.apply(weights_init)

        # Resume from checkpoint
        if self.load_checkpoint != "":
            self.RNet.load_state_dict(build_state_dict(self.load_checkpoint))
            print(f'Load from: {self.load_checkpoint}')

    def train(self, word_embs, labels, s_len=None):
        """Training process

        Positional arguments:
        word_embs -- vector representations for all examples
                        if elmo_concat:
                            if not lstm: (954, 3072)
                            if lstm: (954, 30, 3072)
                        if elmo_avg:
                            if not lstm: (954, 1024)
                            if lstm: (954, 30, 1024)
                        if GloVe: (954, GLOVE_DIM)
        labels -- ground truth (954, )
        """
        labels = np.expand_dims(labels, axis=1)
        self.load_network()
        lr = self.lr
        optimizer = optim.Adam(self.RNet.parameters(),
                               lr=lr,
                               betas=(self.cfg.TRAIN.COEFF.BETA_1,
                                      self.cfg.TRAIN.COEFF.BETA_2),
                               eps=self.cfg.TRAIN.COEFF.EPS)
        epoch = self.cfg.TRAIN.START_EPOCH
        count = self.cfg.TRAIN.START_EPOCH*self.cfg.BATCH_ITEM_NUM

        if epoch == 0:
            save_model(self.RNet, epoch, self.model_dir)
        while epoch < self.total_epoch:
            epoch += 1
            start_t = time.time()
            batch_inds = list(BatchSampler(SequentialSampler(word_embs),
                                           batch_size=self.batch_size,
                                           drop_last=True))

            if epoch % self.lr_decay_per_epoch == 0:
                # update learning rate
                lr *= self.cfg.TRAIN.LR_DECAY_RATE
                print(f'learning rate updated: {lr}')
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            for i, inds in enumerate(batch_inds, 0):
                real_labels = labels[inds]
                curr_batch = word_embs[inds]
                seq_lengths = [s_len[ii] for ii in inds]
                sort_idx = sorted(range(len(seq_lengths)), key=lambda k: seq_lengths[k], reverse=True)
                seq_lengths.sort(reverse=True)
                curr_batch = curr_batch[sort_idx]
                curr_batch = curr_batch[:, :seq_lengths[0], :]
                real_labels = real_labels[sort_idx]

                curr_batch_tensor = Variable(curr_batch, requires_grad=True)
                real_label_tensor = Variable(torch.from_numpy(real_labels))

                # real_seq_len = seq_lengths.copy()
                # seq_lengths[0] = self.cfg.LSTM.SEQ_LEN
                pack = pack_padded_sequence(curr_batch_tensor, seq_lengths, batch_first=True)
                output_scores = self.RNet(pack, len(seq_lengths), seq_lengths)

                optimizer.zero_grad()
                loss_func = nn.MSELoss()
                loss = loss_func(output_scores, real_label_tensor.type(torch.FloatTensor))
                loss.backward()
                # gradient clipping, if necessary
                # clip_grad_value_(self.RNet.parameters(), 2)
                plot_grad_flow(self.RNet.named_parameters(), count)
                # plot_grad_flow_v0(self.RNet.named_parameters(), count)
                optimizer.step()

                count += 1
                if i % 10 == 0:
                    write_summary(loss, 'loss', self.summary_writer, count)

            end_t = time.time()
            print(f'[{epoch}/{self.total_epoch}][{i}/{len(batch_inds)-1}] Loss: {loss:.4f}'
                  f' Total Time: {(end_t-start_t):.2f}sec')
            if epoch % 2 == 0 or epoch == 1:
                save_model(self.RNet, epoch, self.model_dir)
        save_model(self.RNet, self.total_epoch, self.model_dir)

    def evaluate(self, word_embs, max_diff, min_value):
        """Make predictions and evaluate the model

        Positional arguments:
        word_embs -- vector representations for all examples
                        if elmo_concat: (408, 3072)
                        if elmo_avg: (408, 1024)
                        if GloVe: (408, GLOVE_DIM)
        max_diff -- for normalization
        min_value -- for normalization
        """
        self.load_network()
        self.RNet.eval()

        num_items = word_embs.shape[0]
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
            # output_scores, h = self.RNet(curr_batch)
            output_scores = self.RNet(curr_batch)
            # all_hiddens_list.append(h)
            for curr_score in output_scores.data.tolist():
                rating_lst.append(curr_score[0]*max_diff+min_value)
            count += batch_size
        # all_hiddens = torch.cat(tuple(all_hiddens_list)).data.numpy()
        return np.array(rating_lst)  # , all_hiddens

    def analyze(self):
        """Analyze weights"""
        self.load_network()
        self.RNet.eval()

        w = self.RNet.conv1.weight.data.numpy()
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


# Elmo
def get_sentence_elmo(s, embedder, elmo_mode='concat', LSTM=False, seq_len=None):
    """Get ELMo vector representation for each sentence"""
    s = s.replace('\'ve', ' \'ve')
    s = s.replace('\'re', ' \'re')
    s = s.replace('\'ll', ' \'ll')
    s = s.replace('n\'t', ' n\'t')
    s = s.replace('\'d', ' \'d')
    s = s.replace('-', ' ')
    s = s.replace('\'s', ' \'s')
    modified_s = re.sub('#', '.', s).strip('.').split('.')
    modified_s = list(filter(None, modified_s))
    raw_tokens = ['<bos>']
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
    raw_tokens.append('<eos>')
    expected_embedding = embedder.embed_sentence(raw_tokens)  # [3, actual_sentence_len+2, 1024]
    if not LSTM:
        expected_embedding = np.mean(expected_embedding, axis=1)    # averaging on # of words
        if elmo_mode == 'concat':
            expected_embedding = np.concatenate(expected_embedding)
        elif elmo_mode == 'avg':
            expected_embedding = np.mean(expected_embedding, axis=0)
    else:
        sentence_len = expected_embedding.shape[1]
        if elmo_mode == 'concat':
            # [seq_len, 3024]
            expected_embedding = np.concatenate(expected_embedding, axis=1)
        elif elmo_mode == 'avg':
            expected_embedding = np.mean(expected_embedding, axis=0)
        # chop/pad
        expected_embedding_padded, sl = padded(expected_embedding, seq_len)
    expected_embedding_tensor = torch.from_numpy(expected_embedding_padded)
    return expected_embedding_tensor, sl


def padded(emb, seq_len):
    sentence_len, dim = emb.shape  # sentence_len = actual_sentence_len + 2
    result = np.zeros((seq_len, dim))
    l = seq_len
    if seq_len <= sentence_len:
        result[:seq_len-1, :] = emb[:seq_len-1, :]
        result[-1, :] = emb[-1, :]  # <eos>
    else:
        result[:sentence_len, :] = emb[:sentence_len, :]
        # result[sentence_len:, :] = np.random.rand((seq_len - sentence_len), dim)
        result[sentence_len:, :] = 0
        l = sentence_len
    return result, l


def plot_grad_flow(named_parameters, global_step):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    print('------------\n', global_step)
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
            print(n, ': ', p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow at step " + format(global_step))
    plt.grid(True)
    plt.legend([mlines.Line2D([0], [0], color="c", lw=4),
                mlines.Line2D([0], [0], color="b", lw=4),
                mlines.Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig(IMG_DIR + format(global_step), bbox_inches='tight')
    plt.close()


def plot_grad_flow_v0(named_parameters, count):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    print("plotting gradient flow at step " + format(count))
    if count == 27:
        plt.savefig(IMG_DIR + "epoch0_" + format(count), bbox_inches='tight')
        plt.close()


def main():
    result = get_sentence("Good Morning!")
    print(result)

if __name__ == "__main__":
    main()
