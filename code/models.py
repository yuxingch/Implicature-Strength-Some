from itertools import combinations
import logging
import os
import re
import ssl
import sys
import time

from allennlp.commands.elmo import ElmoEmbedder
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchtext.vocab as vocab
from torch.utils.data.sampler import SequentialSampler, BatchSampler, RandomSampler
from torch.nn.utils import clip_grad_value_
from torch.nn.utils.rnn import pack_padded_sequence

from utils import mkdir_p, weights_init, save_model
ssl._create_default_https_context = ssl._create_unverified_context


#logging.basicConfig(level=logging.INFO)

OPTION_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/" \
              "2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
WEIGHT_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/" \
              "2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

BERT_DIM = 768
BERT_LARGE_DIM = 1024
GLOVE_DIM = 100
ELMO_DIM = 1024
glove = vocab.GloVe(name='6B', dim=GLOVE_DIM)

torch.manual_seed(1)

_UNK = torch.randn(GLOVE_DIM,)
_PAD = torch.randn(GLOVE_DIM,)
_BOS = torch.randn(GLOVE_DIM,)
_EOS = torch.randn(GLOVE_DIM,)


def build_state_dict(config_net):
    """Build dictionary to store the state of our neural net"""
    return torch.load(config_net, map_location=lambda storage, loc: storage)['state_dict']


class RatingModel(object):

    def __init__(self, cfg, output_dir):
        """Intialize RatingModel

        Positional arguments:
        cfg -- configuration dictionary
        output_dir -- path to save checkpoints and logs
        """
        self.cfg = cfg
        if self.cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.best_model_dir = os.path.join(output_dir, 'Best Model')
            mkdir_p(self.model_dir)
            mkdir_p(self.best_model_dir)

        self.batch_size = self.cfg.TRAIN.BATCH_SIZE
        self.total_epoch = self.cfg.TRAIN.TOTAL_EPOCH
        self.load_checkpoint = self.cfg.RESUME_DIR
        self.lr = self.cfg.TRAIN.LR
        self.lr_decay_per_epoch = self.cfg.TRAIN.LR_DECAY_EPOCH
        self.dropout = [self.cfg.TRAIN.DROPOUT.FC_1, self.cfg.TRAIN.DROPOUT.FC_2]
        self.drop_prob = self.cfg.LSTM.DROP_PROB
        self.interval = self.cfg.TRAIN.INTERVAL
        self.loss_func = nn.MSELoss()

        self.train_loss_history = []
        self.val_loss_history = []
        self.val_r_history = []

        self.best_val_loss = float("inf")
        self.best_val_r = 0.
        self.best_val_epoch = 0

        # gpu
        self.gpus = []
        for i in range(self.cfg.GPU_NUM):
            self.gpus.append(i)
        if self.cfg.CUDA:
            torch.cuda.set_device(self.gpus[0])
            cudnn.benchmark = True
            self.loss_func.cuda()

    def load_network(self):
        """Initialize the network or load from checkpoint"""
        from net import RateNet, RateNet2D, BiLSTM, BiLSTMAttn
        logging.info('initializing neural net')

        self.RNet = None
        vec_dim = GLOVE_DIM
        if self.cfg.IS_ELMO:
            vec_dim = ELMO_DIM
        elif self.cfg.IS_BERT:
            vec_dim = BERT_LARGE_DIM if self.cfg.BERT_LARGE else BERT_DIM
        if self.cfg.LSTM.FLAG:
            if self.cfg.LSTM.ATTN:
                self.RNet = BiLSTMAttn(vec_dim, self.cfg.LSTM.SEQ_LEN,
                                       self.cfg.LSTM.HIDDEN_DIM,
                                       self.cfg.LSTM.LAYERS,
                                       self.drop_prob, self.dropout,
                                       self.cfg.LSTM.BIDIRECTION,
                                       self.cfg.CUDA)
            else:
                self.RNet = BiLSTM(vec_dim, self.cfg.LSTM.SEQ_LEN,
                                   self.cfg.LSTM.HIDDEN_DIM,
                                   self.cfg.LSTM.LAYERS,
                                   self.drop_prob, self.dropout,
                                   self.cfg.LSTM.BIDIRECTION, self.cfg.CUDA)
        else:
            self.RNet = RateNet(vec_dim, self.dropout)
        self.RNet.apply(weights_init)

        # Resume from checkpoint
        if self.load_checkpoint != "":
            self.RNet.load_state_dict(build_state_dict(self.load_checkpoint))
            logging.info(f'Load from: {self.load_checkpoint}')

    def train(self, X, y, L):
        """Training process

        Positional arguments:
        X -- dict(), keys = ["train", "val"]
             vector representations for train/val examples
        y -- dict(), keys = ["train", "val"]
             human judgments for training examples
        L -- dict(), keys = ["train", "val"]
             number of tokens in each training example before
             chopping/padding
        """
        X_train, X_val = X["train"], X["val"]
        y_train, y_val = y["train"], y["val"]
        L_train, L_val = L["train"], L["val"]

        y_train = np.expand_dims(y_train, axis=1)
        self.load_network()
        # gpu
        if self.cfg.CUDA:
            self.RNet.cuda()
        lr = self.lr
        optimizer = optim.Adam(self.RNet.parameters(),
                               lr=lr,
                               betas=(self.cfg.TRAIN.COEFF.BETA_1,
                                      self.cfg.TRAIN.COEFF.BETA_2),
                               eps=self.cfg.TRAIN.COEFF.EPS)
        epoch = self.cfg.TRAIN.START_EPOCH
        count = self.cfg.TRAIN.START_EPOCH*self.cfg.BATCH_ITEM_NUM

        count_loss = []
        if epoch == 0:
            # Purely random
            save_model(self.RNet, epoch, self.model_dir)
        while epoch < self.total_epoch:
            epoch += 1
            start_t = time.time()
            batch_inds = list(BatchSampler(RandomSampler(X_train),
                                           batch_size=self.batch_size,
                                           drop_last=False))

            if epoch % self.lr_decay_per_epoch == 0:
                # update learning rate
                lr = self.lr * (self.cfg.TRAIN.LR_DECAY_RATE ** (epoch / self.lr_decay_per_epoch))
                logging.info(f'learning rate updated: {lr}')
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            total_loss = 0
            for i, inds in enumerate(batch_inds, 0):
                y_batch = y_train[inds]
                X_batch = X_train[inds]
                seq_lengths = [L_train[ii] for ii in inds]

                sort_idx = sorted(range(len(seq_lengths)), key=lambda k: seq_lengths[k], reverse=True)
                seq_lengths.sort(reverse=True)
                X_batch = X_batch[sort_idx].float()
                y_batch = y_batch[sort_idx]
                y_batch = torch.from_numpy(y_batch).float()

                if self.cfg.CUDA:
                    X_batch = X_batch.cuda()
                    y_batch = y_batch.cuda()

                if self.cfg.LSTM.FLAG:
                    pack = pack_padded_sequence(X_batch, seq_lengths,
                                                batch_first=True)
                    output_scores, _ = self.RNet(pack, len(seq_lengths), seq_lengths)
                else:
                    output_scores, _ = self.RNet(X_batch)
                optimizer.zero_grad()
                loss = self.loss_func(output_scores, y_batch)
                total_loss += loss.item()
                loss.backward()

                clip_grad_value_(self.RNet.parameters(), 2)
                optimizer.step()

                count += 1
                if count % 3 == 0 or count == 1:
                    count_loss.append((count, loss))
            end_t = time.time()

            # validation
            if X_val is not None:
                val_loss, val_r = self.validation(X_val, y_val, L_val)
                self.RNet.train()   # reset to train mode
                if val_r > self.best_val_r:
                    self.best_val_r = val_r
                    self.best_val_loss = val_loss
                    self.best_val_epoch = epoch
                    # save current best
                    # save_model(self.RNet, epoch, self.best_model_dir)
                self.val_loss_history.append(val_loss)
                self.val_r_history.append(val_r)
            else:
                val_loss = 0
                val_r = 0
            self.train_loss_history.append(total_loss)

            logging.info(f'[{epoch}/{self.total_epoch}][{i+1}/{len(batch_inds)}]'
                         f' total train loss: {total_loss:.4f}; total val loss: {val_loss:.4f}'
                         f' val r: {val_r:.4f}; time: {(end_t-start_t):.2f}sec')

            if epoch % self.interval == 0 or epoch == 1:
                count_loss = []
                save_model(self.RNet, epoch, self.model_dir)
        # save checkpoint for the last epoch
        save_model(self.RNet, self.total_epoch, self.model_dir)
        logging.info(f'Best epoch {self.best_val_epoch} with val_r = {self.best_val_r:.4f}.')

    def validation(self, X_val, y_val, L_val=None):
        self.RNet.eval()
        batch_inds = list(BatchSampler(RandomSampler(X_val),
                                       batch_size=self.batch_size,
                                       drop_last=False))
        total_val_loss = 0
        y_preds_lst = []
        val_inds = []
        with torch.no_grad():
            for i, inds in enumerate(batch_inds, 0):
                val_inds += inds
                y_batch = y_val[inds]
                X_batch = X_val[inds]
                seq_lengths = [L_val[ii] for ii in inds]

                sort_idx = sorted(range(len(seq_lengths)), key=lambda k: seq_lengths[k], reverse=True)
                seq_lengths.sort(reverse=True)
                X_batch = X_batch[sort_idx].float()
                y_batch = y_batch[sort_idx]
                y_batch = torch.from_numpy(y_batch).float()

                if self.cfg.CUDA:
                    X_batch = X_batch.cuda()
                    y_batch = y_batch.cuda()

                if self.cfg.LSTM.FLAG:
                    pack = pack_padded_sequence(X_batch, seq_lengths,
                                                batch_first=True)
                    output_scores, _ = self.RNet(pack, len(seq_lengths),
                                                 seq_lengths)
                else:
                    output_scores, _ = self.RNet(X_batch)

                loss = self.loss_func(output_scores.squeeze(), y_batch)
                total_val_loss += loss.item()

                output_scores = output_scores.data.tolist()

                temp_rating = [0]*len(sort_idx)
                cnt = 0
                for s in sort_idx:
                    temp_rating[s] = output_scores[cnt][0]
                    cnt += 1
                for curr_score in temp_rating:
                    y_preds_lst.append(curr_score*(self.cfg.MAX_VALUE - self.cfg.MIN_VALUE) + self.cfg.MIN_VALUE)
        y_val = y_val[val_inds]
        val_coeff = np.corrcoef(np.array(y_preds_lst), np.array(y_val))[0, 1]
        return total_val_loss, val_coeff

    def evaluate(self, X, max_diff, min_value, sl):
        """Make predictions and evaluate the model

        Positional arguments:
        X -- vector representations for all examples
        max_diff -- for normalization
        min_value -- for normalization
        sl -- length of the sequence
        """
        self.load_network()
        self.RNet.eval()

        # gpu
        if self.cfg.CUDA:
            self.RNet.cuda()

        num_items = X.shape[0]
        batch_size = min(num_items, self.batch_size)

        rating_lst = []
        count = 0
        all_hiddens_list = []
        all_attn = np.zeros((num_items, self.cfg.LSTM.SEQ_LEN, 1))
        diff = 0
        while count < num_items:
            iend = count + batch_size
            if iend > num_items:
                diff = iend - num_items
                iend = num_items
                # break
                #count = num_items - batch_size
            X_batch = X[count:iend]
            seq_lengths = sl[count:iend]

            sort_idx = sorted(range(len(seq_lengths)), key=lambda k: seq_lengths[k], reverse=True)
            seq_lengths.sort(reverse=True)
            max_seq_len_batch = seq_lengths[0]
            X_batch = X_batch[sort_idx]
            X_batch = X_batch[:, :max_seq_len_batch, :]

            if self.cfg.CUDA:
                X_batch = X_batch.float().cuda()

            if self.cfg.LSTM.FLAG:
                pack = pack_padded_sequence(X_batch, seq_lengths, batch_first=True)
                output_scores, attn_weights = self.RNet(pack, len(seq_lengths), seq_lengths)
            else:
                output_scores, attn_weights = self.RNet(X_batch)
            output_scores = output_scores.data.tolist()

            temp_rating = [0]*len(sort_idx)
            cnt = 0
            if attn_weights is not None:
                revert_attn_weights = np.zeros(attn_weights.shape)  # (batch_size, 8, seq_len, seq_len)
            for s in sort_idx:
                temp_rating[s] = output_scores[cnt][0]
                if attn_weights is not None:
                    revert_attn_weights[s, :, :] = attn_weights[cnt, :, :]
                cnt += 1
            #temp_rating = temp_rating[diff:]
            if attn_weights is not None:
                all_attn[count:iend, :max_seq_len_batch, :] = revert_attn_weights[:, :, :]
            for curr_score in temp_rating:
                rating_lst.append(curr_score*max_diff+min_value)
            count += batch_size
        return np.array(rating_lst), all_attn


#####################################
# Helper functions for word vectors #
#####################################
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


def get_sentence_glove(s, LSTM=False, not_contextual=True, seq_len=30):
    raw_tokens = preprocess_utterance(s)
    lst = []
    if LSTM:
        lst = [_BOS]
    for w in raw_tokens:
        curr_emb = get_word(w.lower())
        if torch.all(torch.eq(curr_emb, _UNK)):
            continue
        else:
            lst.append(curr_emb)
    if LSTM:
        lst.append(_EOS)
    all_embs = torch.stack(lst)
    if not LSTM:
        return torch.mean(all_embs, 0), len(raw_tokens)
    if not not_contextual:
        expected_embedding_padded, sl = context_padded(all_embs, seq_len)
    else:
        expected_embedding_padded, sl = padded(all_embs, seq_len)
    assert sl <= seq_len
    expected_embedding_tensor = torch.from_numpy(expected_embedding_padded)
    return expected_embedding_tensor, sl


def preprocess_utterance(s, split_off_clitics=True,max_utterances=None):
  if split_off_clitics:
    s = s.replace('\'ve', ' \'ve')
    s = s.replace('\'re', ' \'re')
    s = s.replace('\'ll', ' \'ll')
    s = s.replace('n\'t', ' n\'t')
    s = s.replace('\'d', ' \'d')
    s = s.replace('-', ' ')
    s = s.replace('\'s', ' \'s')
  modified_s = re.sub('#', '.', s).strip('.').split('.')
  modified_s = list(filter(None, modified_s))
  if max_utterances and max_utterances < len(modified_s):
    modified_s = modified_s[-max_utterances]
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
  
  return raw_tokens


def tokenizer(s, pad_symbol=True, seq_len=None, from_right=True):
    """If `pad_symbol=True`, pad <bos> at the beginning and <eos> at the end"""
    
    if pad_symbol:
        raw_tokens = ['<S>']
    else:
        raw_tokens = []
        
    raw_tokens.extend(preprocess_utterance(s))
    
    if pad_symbol:
        raw_tokens.append('</S>')
    total_len = len(raw_tokens)
    if seq_len and seq_len-2 < total_len:
        seq_len -= 2
        if from_right:
            return raw_tokens[:seq_len]
        else:
            total_len = len(raw_tokens)
            return raw_tokens[total_len-seq_len:]
    return raw_tokens


# Elmo
def get_sentence_elmo(s, c, embedder, layer=2, not_contextual=True, LSTM=False, seq_len=None):
    """Get ELMo vector representation for each sentence"""

    if not not_contextual:
      s = s + " </S> <S> " + c
    if not LSTM:
        raw_tokens = tokenizer(s, pad_symbol=False)
        expected_embedding = embedder.embed_sentence(raw_tokens)
        sl = seq_len
        expected_embedding = np.mean(expected_embedding, axis=1)    # averaging on # of words
        expected_embedding = expected_embedding[layer, :].squeeze()
    else:
        raw_tokens = tokenizer(s)
        expected_embedding = embedder.embed_sentence(raw_tokens)  # [3, actual_sentence_len+2, 1024]
        sentence_len = expected_embedding.shape[1]
        expected_embedding = expected_embedding[layer, :, :].squeeze()
        # chop/pad
        if not_contextual:
            expected_embedding_padded, sl = padded(expected_embedding, seq_len)
        else:
            expected_embedding_padded, sl = context_padded(expected_embedding, seq_len)
        assert sl <= seq_len
    expected_embedding_tensor = torch.from_numpy(expected_embedding_padded)
    # expected_embedding_tensor = torch.from_numpy(expected_embedding)
    return expected_embedding_tensor, sl

# BERT from huggingface models
def get_sentence_bert(s, bert_tokenizer, bert_model, layer = 11, GPU=False, LSTM=False, max_seq_len=None, is_single=True):
    s = " ".join(preprocess_utterance(s, split_off_clitics = False))
    s = "[CLS] " + s + " [SEP]"
    tokenized_text = bert_tokenizer.tokenize(s)
    indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0] * len(indexed_tokens)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    bert_output = torch.zeros((1,max_seq_len, bert_model.config.hidden_size))
    if GPU:
      tokens_tensor = tokens_tensor.cuda()
      segments_tensors = segments_tensors.cuda()
      bert_output = bert_output.cuda()
      
    sl = min(len(indexed_tokens), max_seq_len)

    with torch.no_grad():
        outputs = bert_model(tokens_tensor, token_type_ids=segments_tensors)
        bert_output[:, :sl, :] = outputs[2][layer][:,:sl, :]
        bert_output = bert_output.squeeze()  # (max_seq_len, 768)

    if GPU:
      bert_output = bert_output.cpu()

    if LSTM:
        return bert_output, sl
    else:
        bert_mean = torch.mean(bert_output, axis=0)
        return bert_mean, sl

def get_sentence_bert_context(s, c, bert_tokenizer, bert_model, layer = 11,
                              GPU=False, LSTM=False, max_sentence_len=None, 
                              max_context_len=None, max_context_utterances=None):
    s = " ".join(preprocess_utterance(s, split_off_clitics = False))
    s = " ".join(preprocess_utterance(s, split_off_clitics = False, max_utterances=max_context_utterances))
    s = "[CLS]" + s + " [SEP] " + c + " [SEP]" 
    tokenized_text = bert_tokenizer.tokenize(s)
    indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)
    if len(indexed_tokens) > max_sentence_len + max_context_len:
      indexed_tokens = indexed_tokens[:(max_sentence_len + max_context_len)]
    s_len = tokenized_text.index("[SEP]")
    segments_ids = [0] * (s_len + 1) + [1] * (len(indexed_tokens) - s_len - 1)
    
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    bert_output = torch.zeros((1,max_sentence_len, bert_model.config.hidden_size))
    if GPU:
      tokens_tensor = tokens_tensor.cuda()
      segments_tensors = segments_tensors.cuda()
      bert_output = bert_output.cuda()
      
    sl = min(s_len, max_sentence_len)

    with torch.no_grad():
        outputs = bert_model(tokens_tensor, token_type_ids=segments_tensors)
        bert_output[:, :sl, :] = outputs[2][layer][:,:sl, :]
        bert_output = bert_output.squeeze()  # (max_seq_len, 768)

    if GPU:
      bert_output = bert_output.cpu()

    if LSTM:
        return bert_output, sl
    else:
        bert_mean = torch.mean(bert_output, axis=0)
        return bert_mean, sl


def padded(emb, seq_len):
    sentence_len, dim = emb.shape  # sentence_len = actual_sentence_len + 2
    result = np.zeros((seq_len, dim))
    l = seq_len
    if seq_len <= sentence_len:
        result[:seq_len-1, :] = emb[:seq_len-1, :]
        result[-1, :] = emb[-1, :]  # <eos>
    else:
        result[:sentence_len, :] = emb[:sentence_len, :]
        # If want to use random values instead of zero, un-comment the line below:
        # result[sentence_len:, :] = np.random.rand((seq_len - sentence_len), dim)
        result[sentence_len:, :] = 0
        l = sentence_len
    return result, l


def context_padded(emb, seq_len):
    context_len, dim = emb.shape
    result = np.zeros((seq_len, dim))
    l = seq_len
    if seq_len <= context_len:
        result[0, :] = emb[0, :]    # <bos>
        result[1:, :] = emb[context_len-seq_len+1:, :]
    else:
        result[:context_len, :] = emb[:context_len, :]
        result[context_len:, :] = 0
        l = context_len
    return result, l


def main():
    result, _ = get_sentence_glove("This is a dummy test utterance.")
    print(result)

if __name__ == "__main__":
    main()
