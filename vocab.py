import torchtext.vocab as vocab
import pandas as pd
import re
import numpy as np
from tqdm import tqdm
from itertools import combinations
import matplotlib.pyplot as plt


def split_by_whitespace(sentence):
    words = []
    for space_separated_fragment in sentence.strip('.').strip().split():
        words.extend(re.split(" ", space_separated_fragment))
    return [w.lower() for w in words if w]


def get_words_per_paragraph(p):
    modified_p = re.sub('#', '.', p).strip('.').split('.')
    modified_p = list(filter(None, modified_p))
    curr_words = []
    for s in modified_p:
        s = re.sub('[^a-zA-Z0-9 \n\.]', '', s)
        curr_words += split_by_whitespace(s)
    return curr_words


def build_dictionary(input):
    input_df = pd.read_csv(input, sep='\t')
    dict_item_paragraph_raw = input_df[['Item_ID', '20-b']].groupby('Item_ID')['20-b'].apply(list).to_dict()
    all_paragraphs = []
    for (k, v) in dict_item_paragraph_raw.items():
        all_paragraphs.append(dict_item_paragraph_raw[k][0])
    # print(all_paragraphs[:10])
    d = set()
    for p in all_paragraphs:
        curr_words = get_words_per_paragraph(p)
        d = d.union(curr_words)
    return d


def prune_glove(glossary, glove_dim=100):
    glove_path = 'glove.6B.'+str(glove_dim)+'d.txt'
    vocab_size = int(len(glossary))
    emb_value = np.zeros((vocab_size, glove_dim))
    word2id = {}
    id2word = {}
    idx = 0
    with open(glove_path, 'r') as fh:
        for line in tqdm(fh, total=vocab_size):
            line = line.lstrip().rstrip().split(" ")
            word = line[0]
            if word in glossary:
                vector = list(map(float, line[1:]))
                # value = list(map(float, line[1:]))[at_dim]
                emb_value[idx, :] = vector
                word2id[word] = idx
                id2word[idx] = word
                idx += 1
            else:
                print(word)
    # print(idx, len(word2id))
    pruned = np.zeros((idx, glove_dim))
    pruned[:idx, :] = emb_value[:idx, :]
    # print(pruned[11], id2word[11], word2id['said'])
    return pruned, word2id, id2word


def save_pruned_glove(pruned, id2word):
    n, d = pruned.shape
    save_path = './pruned_glove.6B.'+str(d)+'d.txt'
    f = open(save_path, 'w')
    for i in range(n):
        curr_vector = pruned[i, :]
        curr_word = id2word[i]
        temp = ''
        for j in range(d):
            temp += ' ' + format(curr_vector[j])
        f.write(curr_word)
        f.write(temp+"\n")
    f.close()


def load_prune_glove(glove_path, glove_dim=100, vocab_size=7269):
    # glove_path = 'pruned_glove.6B.100d.txt'
    emb_value = np.zeros((vocab_size, glove_dim))
    word2id = {}
    id2word = {}
    idx = 0
    with open(glove_path, 'r') as fh:
        for line in tqdm(fh, total=vocab_size):
            line = line.lstrip().rstrip().split(" ")
            word = line[0]
            vector = list(map(float, line[1:]))
            # value = list(map(float, line[1:]))[at_dim]
            emb_value[idx, :] = vector
            word2id[word] = idx
            id2word[idx] = word
            idx += 1
    return emb_value, word2id, id2word


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2)))


def find_closest(word, id, emb):
    combinations_ids = combinations(range(100), 3)
    l = 161700
    # l = 3921225
    curr_max = -1
    max_cb = None
    for cb in tqdm(combinations_ids, total=l):
        cb_list = list(cb)
        prune_emb = np.zeros_like(emb)
        prune_emb[cb_list] = emb[cb_list]
        cos = cosine_similarity(prune_emb, emb)
        if cos > curr_max:
            curr_max = cos
            max_cb = cb_list
    return curr_max, max_cb


def find_closest_for_sentence(emb):
    # f='./embs_single_rand_unk.npy'
    # sentence_embs = np.load(f)
    combinations_ids = combinations(range(100), 3)
    # l = 161700
    curr_max = -1
    max_cb = None
    # for cb in tqdm(combinations_ids, total=l):
    for cb in combinations_ids:
        cb_list = list(cb)
        prune_emb = np.zeros_like(emb)
        prune_emb[cb_list] = emb[cb_list]
        cos = cosine_similarity(prune_emb, emb)
        # print(cos, curr_max)
        if cos > curr_max:
            curr_max = cos
            max_cb = cb_list
    return curr_max, max_cb


def plot_frequency():
    input_df = pd.read_csv('./most_important_dims_paragraph.csv', sep='\t')
    lst = input_df.indices.tolist()
    frequency = np.zeros(100)
    for l in lst:
        curr = l.split(',')
        curr = list(map(int, curr))
        frequency[curr] += 1
    # n, bins, patches = plt.hist(frequency, 100, density=True, facecolor='g', alpha=0.75)

    plt.plot(frequency, 'g.', markersize=6)
    plt.xlabel('dimension')
    plt.ylabel('frequency')
    plt.title('Most important dimensions evaluated using cosine similarity')
    plt.grid(True)
    plt.show()


def main():
    # result = build_dictionary('./swbdext.csv')
    # pruned, word2id, id2word = prune_glove(result)
    # save_pruned_glove(pruned, id2word)

    # word_embs, word2id, id2word = load_prune_glove('pruned_glove.6B.100d.txt')
    # # print(word_embs[11], id2word[11], word2id['said'])
    # save_path = './most_important_dims_paragraph.csv'
    # f = open(save_path, 'w')
    # headline = 'n\tsimilarity\tindices\n'
    # f.write(headline)
    # sentence_embs = np.load('./embs_paragraph_rand_unk_2.npy')
    # # for idx in range(1362):
    # for idx in tqdm(range(1951), total=1951):
    #     curr_max, max_cb = find_closest_for_sentence(sentence_embs[idx])
    #     s = format(idx)
    #     s += '\t'+format(curr_max)+'\t'
    #     s += ','.join(str(e) for e in max_cb)
    #     s += '\n'
    #     f.write(s)
    # f.close()
    plot_frequency()


if __name__ == "__main__":
    main()
