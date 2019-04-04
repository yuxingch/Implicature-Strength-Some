import torch
import torch.nn as nn


def fc_layer(in_features, out_features, dropout):
    return nn.Sequential(nn.Linear(in_features, out_features, bias=True),
                         nn.BatchNorm1d(out_features),
                         nn.ReLU(True),
                         nn.Dropout(p=dropout))


class RateNet(nn.Module):
    def __init__(self, glove_dim, dropout, plus_dim=0):
        super(RateNet, self).__init__()
        self.glove_dim = glove_dim
        self.plus_dim = plus_dim
        self.input_dim = self.glove_dim + self.plus_dim
        self.fc1, self.fc2 = None, None
        self.get_score = None
        self.dropout = dropout
        self.define_module()

    def define_module(self):
        self.fc1 = fc_layer(self.input_dim, 64, self.dropout[0])
        self.fc2 = fc_layer(64, 32, self.dropout[1])

        self.get_score = nn.Sequential(
            nn.Linear(32, 1, bias=True))

    def forward(self, word_embs):
        h1 = self.fc1(word_embs)
        h = self.fc2(h1)
        # return self.get_score(h), h1
        return self.get_score(h)


class RateNetELMo(nn.Module):
    def __init__(self, glove_dim, dropout, plus_dim=0):
        super(RateNetELMo, self).__init__()
        self.glove_dim = glove_dim
        self.plus_dim = plus_dim
        self.input_dim = self.glove_dim + self.plus_dim
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
        h1 = self.fc1(word_embs)
        h = self.fc2(h1)
        return self.get_score(h)


# TODO:
class RateNet2D(nn.Module):
    def __init__(self, glove_dim, plus_dim=0):
        super(RateNet2D, self).__init__()
        self.glove_dim = glove_dim
        self.plus_dim = plus_dim
        self.input_dim = self.glove_dim + self.plus_dim
        # self.fc1, self.fc2 = None, None
        # self.get_score = None
        # self.define_module()
        self.conv1 = nn.Conv1d(100, 64, 4, 2, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv1d(64, 128, 4, 2, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(True)
        self.conv3 = nn.Conv1d(128, 256, 4, 2, 1)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU(True)
        self.conv4 = nn.Conv1d(256, 512, 4, 2, 1)
        self.bn4 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU(True)
        self.conv5 = nn.Conv1d(512, 1024, 4, 2, 1)
        self.bn5 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU(True)
        self.define_module()
        # self.conv1 = nn.Conv1d(8, 4, 4, 2, 1)
        # self.bn1 = nn.BatchNorm1d(64)
        # self.relu = nn.ReLU(True)
        # self.conv2 = nn.Conv1d(4, 2, 4, 2, 1)
        # self.bn2 = nn.BatchNorm1d(32)
        # self.relu = nn.ReLU(True)
        # self.conv1 = nn.Conv1d(100, 64, 4, 2, 1)
        # self.bn1 = nn.BatchNorm1d(64)
        # self.relu = nn.ReLU(True)
        # self.conv2 = nn.Conv1d(64, 32, 4, 2, 1)
        # self.bn2 = nn.BatchNorm1d(32)
        # self.relu = nn.ReLU(True)

    def define_module(self):
        self.get_score = nn.Sequential(
            nn.Linear(1024, 1, bias=True))

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


# Binary LSTM model
class BiLSTMELMo(nn.Module):
    """
    The purpose of this module is to encode a sequence (sentence/paragraph)
    using a bidirectional LSTM. It feeds the input through LSTM and returns
    all the hidden states.

    Then, the hidden states will be fed into a fully connected layer to get
    a downward projection, which will be the input of the prediction layer.
    """
    def __init__(self, elmo_dim, seq_len, hidden_dim, num_layers, drop_prob, dropout):
        super(BiLSTMELMo, self).__init__()
        self.elmo_dim = elmo_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.drop_prob = drop_prob
        self.dropout = dropout
        self.define_module()

    def define_module(self):
        self.lstm = nn.LSTM(self.elmo_dim,
                            self.hidden_dim,
                            self.num_layers,
                            batch_first=True,
                            dropout=self.drop_prob,
                            bidirectional=True)
        self.fc = nn.Linear(self.seq_len, 1, bias=True)
        self.fc1 = fc_layer(self.hidden_dim*2, self.hidden_dim, self.dropout[0])
        self.fc2 = fc_layer(self.hidden_dim, self.hidden_dim//2, self.dropout[1])
        self.get_score = nn.Sequential(
            nn.Linear(self.hidden_dim//2, 1, bias=True))

    def forward(self, x, batch_size):
        """

        x - Tensor shape (batch_size, seq_len, input_size)
                we need to permute the first and the second axis

        output - Tensor shape (batch_size, 1)
        """
        h0 = torch.randn(self.num_layers*2, batch_size, self.hidden_dim)
        c0 = torch.randn(self.num_layers*2, batch_size, self.hidden_dim)
        x, _ = self.lstm(x, (h0, c0))
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x.reshape(batch_size, self.seq_len, self.hidden_dim*2)
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        x = x.squeeze()
        x = self.fc1(x)
        x = self.fc2(x)
        return self.get_score(x)
