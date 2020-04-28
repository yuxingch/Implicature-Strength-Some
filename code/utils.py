import os
import errno
from copy import deepcopy
import string

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import init
import torch.nn as nn
from tqdm import tqdm


def mkdir_p(path):
    """Create a directory if not exist"""
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
    return


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif classname.find('LSTM') != -1:
        # https://discuss.pytorch.org/t/initializing-rnn-gru-and-lstm-correctly/23605
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)


def save_model(RNet, epoch, model_dir):
    torch.save(
        {'epoch': epoch,
         'state_dict': RNet.state_dict()},
        '%s/RNet_epoch_%d.pth' % (model_dir, epoch)
    )
    print(f'Save model to {model_dir}')
