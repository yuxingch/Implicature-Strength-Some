import math

import torch
import torch.nn as nn


def fc_layer(in_features, out_features, dropout):
    return nn.Sequential(nn.Linear(in_features, out_features, bias=True),
                         nn.BatchNorm1d(out_features),
                         nn.ReLU(True),
                         nn.Dropout(p=dropout))


class RateNet(nn.Module):

    def __init__(self, emb_dim, dropout):
        super(RateNet, self).__init__()
        self.input_dim = emb_dim
        self.fc1, self.fc2 = None, None
        self.get_score = None
        self.dropout = dropout
        self.define_module()

    def define_module(self):
        self.fc1 = fc_layer(self.input_dim, self.input_dim//2, self.dropout[0])
        self.fc2 = fc_layer(self.input_dim//2, self.input_dim//4, self.dropout[1])
        self.get_score = nn.Sequential(
            nn.Linear(self.input_dim//4, 1, bias=True))

    def forward(self, word_embs):
        h = self.fc1(word_embs)
        h = self.fc2(h)
        return self.get_score(h), None


class RateNet2D(nn.Module):

    def __init__(self, glove_dim):
        super(RateNet2D, self).__init__()
        self.glove_dim = glove_dim
        self.input_dim = self.glove_dim
        self.define_module()

    def define_module(self):
        self.conv1 = nn.Conv1d(100, 64, 4, 2, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv1d(64, 128, 4, 2, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, 4, 2, 1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 512, 4, 2, 1)
        self.bn4 = nn.BatchNorm1d(512)
        self.conv5 = nn.Conv1d(512, 1024, 4, 2, 1)
        self.bn5 = nn.BatchNorm1d(1024)
        self.get_score = nn.Linear(1024, 1, bias=True)

    def forward(self, word_embs):
        h1 = self.conv1(word_embs)
        h = self.bn1(h1)
        h = self.relu(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu(h)
        h = self.conv3(h)
        h = self.bn3(h)
        h = self.relu(h)
        h = self.conv4(h)
        h = self.bn4(h)
        h = self.relu(h)
        h = self.conv5(h)
        h = self.bn5(h)
        h = self.relu(h)
        h = h.view(32, 1024)
        return self.get_score(h)


# (Bi-)LSTM model
class BiLSTM(nn.Module):
    """
    The purpose of this module is to encode a sequence (sentence/paragraph)
    using a bidirectional LSTM. It feeds the input through LSTM and returns
    all the hidden states.

    Then, the hidden states are fed into a projection layer, which in return is
    passed through a sigmoid function to predict the ratings.
    """
    def __init__(self, vec_dim, seq_len, hidden_dim, num_layers, drop_prob, dropout, bidirection, is_gpu, batch_size=32):
        super(BiLSTM, self).__init__()
        self.vec_dim = vec_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.drop_prob = drop_prob
        self.dropout = dropout
        self.bidirect = bidirection
        self.batch_size = batch_size
        self.is_gpu = is_gpu
        self.define_module()

    def define_module(self):
        self.lstm = nn.LSTM(self.vec_dim,
                            self.hidden_dim,
                            self.num_layers,
                            batch_first=True,
                            dropout=self.drop_prob,
                            bidirectional=self.bidirect)
        if self.bidirect:
            self.get_score = nn.Sequential(
                nn.Linear(self.hidden_dim*2, 1, bias=True),
                nn.Sigmoid())
        else:
            self.get_score = nn.Sequential(
                nn.Linear(self.hidden_dim, 1, bias=True),
                nn.Sigmoid())

    def forward(self, x, batch_size, seq_lens):
        """

        x - Tensor shape (curr_batch_size, seq_len, input_size)
                we need to permute the first and the second axis

        output - Tensor shape (curr_batch_size, 1)
        """
#        assert x.shape[0] == batch_size
        if self.bidirect:
            h0 = torch.randn(self.num_layers*2, batch_size, self.hidden_dim)
            c0 = torch.randn(self.num_layers*2, batch_size, self.hidden_dim)
        else:
            h0 = torch.randn(self.num_layers, batch_size, self.hidden_dim)
            c0 = torch.randn(self.num_layers, batch_size, self.hidden_dim)
        if self.is_gpu:
            h0 = h0.float().cuda()
            c0 = c0.float().cuda()
        x, _ = self.lstm(x, (h0, c0))
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        if self.bidirect:
            x = x.reshape(batch_size, seq_lens[0], self.hidden_dim*2)
        else:
            x = x.reshape(batch_size, seq_lens[0], self.hidden_dim)
        x = x.permute(0, 2, 1)
        mask = torch.zeros(x.size())
        if self.bidirect:
          for i in range(batch_size):
            mask[i, :self.hidden_dim, seq_lens[i]-1] = 1
            mask[i, self.hidden_dim:, 0] = 1
        else:
          for i in range(batch_size):
              mask[i, :, seq_lens[i]-1] = 1
        if self.is_gpu:
            mask = mask.cuda()
        x = x * mask  # (batch_size, hidden_dim, max_seq_len)
        x = x.sum(dim=2)  # (batch_size, hidden_dim)
        return self.get_score(x), None


class BiLSTMAttn(nn.Module):
    """
    The purpose of this module is to encode a sequence (sentence/paragraph)
    using a bidirectional LSTM. It feeds the input through LSTM. The LSTM
    output is then passed through a self-attention layer to get a weighted sum.

    Then, the hidden states are fed into a projection layer, which in return is
    passed through a sigmoid function to predict the ratings.
    """
    def __init__(self, vec_dim, seq_len, hidden_dim, num_layers, drop_prob, dropout, bidirection, is_gpu, batch_size=32):
        super(BiLSTMAttn, self).__init__()
        self.vec_dim = vec_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.drop_prob = drop_prob
        self.dropout = dropout
        self.bidirect = bidirection
        self.batch_size = batch_size
        self.is_gpu = is_gpu
        self.define_module()

    def define_module(self):
        self.lstm = nn.LSTM(self.vec_dim,
                            self.hidden_dim,
                            self.num_layers,
                            batch_first=True,
                            dropout=self.drop_prob,
                            bidirectional=self.bidirect)
        if self.bidirect:
            self.attention = SelfAttention(self.hidden_dim*2, self.is_gpu)
            self.get_score = nn.Sequential(
                nn.Linear(self.hidden_dim*2, 1, bias=True),
                nn.Sigmoid())
        else:
            self.attention = SelfAttention(self.hidden_dim, self.is_gpu)
            self.get_score = nn.Sequential(
                nn.Linear(self.hidden_dim, 1, bias=True),
                nn.Sigmoid())

    def forward(self, x, batch_size, seq_lens):
        """

        x - Tensor shape (batch_size, seq_len, input_size)
                we need to permute the first and the second axis

        output - Tensor shape (batch_size, 1)
        """
        if self.bidirect:
            h0 = torch.randn(self.num_layers*2, batch_size, self.hidden_dim)
            c0 = torch.randn(self.num_layers*2, batch_size, self.hidden_dim)
        else:
            h0 = torch.randn(self.num_layers, batch_size, self.hidden_dim)
            c0 = torch.randn(self.num_layers, batch_size, self.hidden_dim)
        if self.is_gpu:
            h0 = h0.float().cuda()
            c0 = c0.float().cuda()
        x, _ = self.lstm(x, (h0, c0))
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        if self.bidirect:
            x = x.reshape(batch_size, seq_lens[0], self.hidden_dim*2)
        else:
            x = x.reshape(batch_size, seq_lens[0], self.hidden_dim)
        x, attn_weights = self.attention(x, seq_lens)
        return self.get_score(x), attn_weights

class SelfAttention(nn.Module):

    def __init__(self, hidden_dim, is_gpu=False):
        super(SelfAttention, self).__init__()
        self.is_gpu = is_gpu
        self.attention1 = nn.Linear(hidden_dim, 50)
        self.attention2 = nn.Linear(50, 1)

    def forward(self, lstm_out, seq_lens):
        # B x max_seq_len x 50
        attention1 = torch.tanh(self.attention1(lstm_out)) 
        # B x max_seq_len x 1
        attention2 = torch.softmax(self.attention2(attention1), dim=1) 
        dot_product = torch.sum(torch.mul(lstm_out, attention2), dim=1)

        attention_weights = attention2
        if self.is_gpu:
          attention_weights = attention_weights.cpu()
        return dot_product, attention_weights.detach().numpy()


#class MultiHeadAttention(nn.Module):
#    def __init__(self, lstm_hidden_size, h, is_gpu=False):
#        super(MultiHeadAttention, self).__init__()
#        assert lstm_hidden_size % h == 0
#        self.num_head = h
#        self.is_gpu = is_gpu
#        self.d_m = lstm_hidden_size
#        self.d_k = self.d_m // self.num_head
#        self.define_module()
#
#    def define_module(self):
#        self.dropout = nn.Dropout(0.1)
#        self.softmax = nn.Softmax(dim=-1)
#        self.linear = nn.Linear(self.d_m, self.d_m)
#        self.attn = None
#
#    def get_mask(self, x, seq_lens):
#        """
#        Get the mask for each padded entry
#        """
#        max_len = seq_lens[0]
#        mask = torch.ones(x.size()[0], max_len)
#        if self.is_gpu:
#            mask = mask.cuda()
#        for idx, curr_l in enumerate(seq_lens):
#            if curr_l < max_len:
#                mask[idx, curr_l:] = 0
#        return mask.unsqueeze(1)
#
#    def attention_func(self, x, mask):
#        assert self.d_k == x.size(-1)
#        mask = mask.unsqueeze(1)
#        scores = torch.matmul(x, x.transpose(-2,-1)) \
#            / math.sqrt(self.d_k)
#        scores = scores.masked_fill(mask==0, -1e9)
#        p_attn = self.softmax(scores)
#        p_attn = self.dropout(p_attn)
#        return torch.matmul(p_attn, x), p_attn
#
#    def forward(self, x, seq_lens):
#        """
#
#        x -- (batch_size, max_seq_len, hidden_dim)
#        self.attn -- (batch_size, num_head, max_seq_len, max_seq_len)
#        """
#        # mask = (x != 0).unsqueeze(-2)
#        curr_batch_size = x.shape[0]
#        mask = self.get_mask(x, seq_lens)
#        x = x.view(curr_batch_size, -1, self.num_head, self.d_k).transpose(1, 2)
#        x, self.attn = self.attention_func(x, mask)
#        x = x.transpose(1, 2).contiguous().view(curr_batch_size, -1, self.num_head*self.d_k)
#        x = self.linear(x)
#        if self.is_gpu:
#            self.attn = self.attn.cpu()
#        return torch.sum(x, 1), self.attn.data.numpy()
#
