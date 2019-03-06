import os
import errno
from copy import deepcopy

from torch.nn import init
import torch
import torch.nn as nn
import torchvision.utils as vutils
from tqdm import tqdm
import numpy as np
import string

import matplotlib.pyplot as plt


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


def save_model(RNet, epoch, model_dir):
    # torch.save(
    #     RNet.state_dict(),
    #     '%s/RNet_epoch_%d.pth' % (model_dir, epoch))
    torch.save(
        {'epoch': epoch,
         'state_dict': RNet.state_dict()},
        '%s/RNet_epoch_%d.pth' % (model_dir, epoch)
    )
    print('Save model')


def find_nearest_words(at_dim, glove_dim=100):
    vocab_size = int(4e5)
    glove_path = './glove.6B.100d.txt'  #./8000.txt'
    emb_value = np.zeros(vocab_size)
    word2id = {}
    id2word = {}
    idx = 0
    with open(glove_path, 'r') as fh:
        for line in tqdm(fh, total=vocab_size):
            line = line.lstrip().rstrip().split(" ")
            word = line[0]
            # vector = list(map(float, line[1:]))
            value = list(map(float, line[1:]))[at_dim]
            emb_value[idx] = value
            word2id[word] = idx
            id2word[idx] = word
            idx += 1

    k = 100
    n = vocab_size
    c = 1
    mean = np.mean(emb_value)
    std = np.std(emb_value)
    centers = np.random.randn(k, c)*std + mean

    centers_old = np.zeros(centers.shape)   # to store old centers
    centers_new = deepcopy(centers)  # Store new centers

    clusters = np.zeros(n)
    distances = np.zeros((n, k))

    error = np.linalg.norm(centers_new - centers_old)
    while error > 0.003:
        # Measure the distance to every center
        for i in range(k):
            distances[:, i] = abs(emb_value - centers_new[i])
        # Assign all training data to closest center

        clusters = np.argmin(distances, axis=1)

        centers_old = deepcopy(centers_new)
        # Calculate mean for every cluster and update the center
        for i in range(k):
            centers_new[i] = np.mean(emb_value[clusters == i], axis=0)
        error = np.linalg.norm(centers_new - centers_old)
        print(error)

    cnt = 0
    for i in range(vocab_size):
        if cnt == 30:
            return
        if clusters[i] == 1:
            print(id2word[i])
            cnt += 1


def order_by_dims(at_dim_max, at_dim_min, glove_dim=100):
    vocab_size = int(4e5)
    glove_path = './glove.6B.100d.txt'  # ./8000.txt'
    emb_value_max = np.zeros((vocab_size, len(at_dim_max)))
    emb_value_min = np.zeros((vocab_size, len(at_dim_min)))
    word2id = {}
    id2word = {}
    idx = 0
    with open(glove_path, 'r') as fh:
        for line in tqdm(fh, total=vocab_size):
            line = line.lstrip().rstrip().split(" ")
            word = line[0]
            # vector = list(map(float, line[1:]))
            i_d = 0
            for d in at_dim_max:
                value = list(map(float, line[1:]))[d]
                emb_value_max[idx, i_d] = value
                i_d += 1
            i_d = 0
            for d in at_dim_min:
                value = list(map(float, line[1:]))[d]
                emb_value_min[idx, i_d] = value
                i_d += 1
            word2id[word] = idx
            id2word[idx] = word
            idx += 1
    frequent_words = ['time', 'person', 'year', 'way', 'day', 'thing', 'man', 'world', 'life', 'hand', 'part', 'child', 'eye', 'woman', 'place', 'work', 'week', 'case', 'point', 'government', 'company', 'number', 'group', 'problem', 'fact', 'be', 'have', 'do', 'say', 'get', 'make', 'go', 'know', 'take', 'see', 'come', 'think', 'look', 'want', 'give', 'use', 'find', 'tell', 'ask', 'work', 'seem', 'feel', 'try', 'leave', 'call', 'good', 'new', 'first', 'last', 'long', 'great', 'little', 'own', 'other', 'old', 'right', 'big', 'high', 'different', 'small', 'large', 'next', 'early', 'young', 'important', 'few', 'public', 'bad', 'same', 'able', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'up', 'about', 'into', 'over', 'after', 'the', 'and', 'a', 'that', 'i', 'it', 'not', 'he', 'as', 'you', 'this', 'but', 'his', 'they', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their']
    # idx = np.random.randint(4e5, size=100)
    idx = [word2id[word] for word in frequent_words]
    chosen_embs_max = emb_value_max[idx, :]
    chosen_embs_min = emb_value_min[idx, :]
    temp = (chosen_embs_max - np.mean(chosen_embs_max, axis=0))
    covariance = 1.0 / 100 * temp.T.dot(temp)
    U,S,V = np.linalg.svd(covariance)
    coord = temp.dot(U[:,0:2])

    for i in range(100):
        plt.text(coord[i, 0], coord[i, 1], id2word[idx[i]], bbox=dict(facecolor='green', alpha=0.1))

    temp = (chosen_embs_min - np.mean(chosen_embs_min, axis=0))
    covariance = 1.0 / 100 * temp.T.dot(temp)
    U,S,V = np.linalg.svd(covariance)
    coord = temp.dot(U[:,0:2])

    for i in range(100):
        plt.text(coord[i, 0], coord[i, 1], id2word[idx[i]], bbox=dict(facecolor='blue', alpha=0.1))

    plt.xlim((np.min(coord[:, 0]), np.max(coord[:, 0])))
    plt.ylim((np.min(coord[:, 1]), np.max(coord[:, 1])))
    plt.show()


def compare_weights(f1='./original.npy', f2='./plus.npy'):
    original = np.load(f1)
    plus = np.load(f2)
    w = plus - original
    w_max, w_min = np.amax(w), np.amin(w)
    w_normalized = (w - w_min) / (w_max - w_min) * 255.0
    plt.imshow(w_normalized, cmap='RdPu')
    plt.colorbar()
    plt.show()


def main():
    # order_by_dims([13, 51, 86, 96])
    # order_by_dims([13, 51, 86, 96], [5, 9, 30, 65])
    compare_weights()


if __name__ == "__main__":
    main()
