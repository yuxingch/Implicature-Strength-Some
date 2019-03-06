import torch.nn as nn


def fc_layer(in_features, out_features):
    return nn.Sequential(nn.Linear(in_features, out_features, bias=True),
                         nn.BatchNorm1d(out_features),
                         nn.ReLU(True),
                         nn.Dropout(p=0.5))


class RateNet(nn.Module):
    def __init__(self, glove_dim, plus_dim=0):
        super(RateNet, self).__init__()
        self.glove_dim = glove_dim
        self.plus_dim = plus_dim
        self.input_dim = self.glove_dim + self.plus_dim
        self.fc1, self.fc2 = None, None
        self.get_score = None
        self.define_module()

    def define_module(self):
        self.fc1 = fc_layer(self.input_dim, 64)
        self.fc2 = fc_layer(64, 32)

        self.get_score = nn.Sequential(
            nn.Linear(32, 1, bias=True))

    def forward(self, word_embs):
        h1 = self.fc1(word_embs)
        h = self.fc2(h1)
        # return self.get_score(h), h1
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


# TODO:
class BiRNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        return
