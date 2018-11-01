import torch.nn as nn


def fc_layer(in_features, out_features):
    return nn.Sequential(nn.Linear(in_features, out_features, bias=True),
                         nn.BatchNorm1d(out_features),
                         nn.ReLU(True),
                         nn.Dropout(p=0.5))


class RateNet(nn.Module):
    def __init__(self, glove_dim):
        super(RateNet, self).__init__()
        self.glove_dim = glove_dim
        self.fc1, self.fc2 = None, None
        self.get_score = None
        self.define_module()

    def define_module(self):
        self.fc1 = fc_layer(self.glove_dim, 64)
        self.fc2 = fc_layer(64, 32)

        self.get_score = nn.Sequential(
            nn.Linear(32, 1, bias=True))
    
    def forward(self, word_embs):
        # print(word_embs.size())
        h = self.fc1(word_embs)
        h = self.fc2(h)
        return self.get_score(h)
        