import numpy as np
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

_PAD = b"<pad>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _UNK]
PAD_ID = 0
UNK_ID = 1


def get_glove(glove_path, glove_dim):
    """Reads from original GloVe .txt file and returns embedding matrix and
    mappings from words to word ids.

    Input:
      glove_path: path to glove.6B.{glove_dim}d.txt
      glove_dim: integer; needs to match the dimension in glove_path

    Returns:
      emb_matrix: Numpy array shape (400002, glove_dim) containing glove embeddings
        (plus PAD and UNK embeddings in first two rows).
        The rows of emb_matrix correspond to the word ids given in word2id and id2word
      word2id: dictionary mapping word (string) to word id (int)
      id2word: dictionary mapping word id (int) to word (string)
    """

    print("Loading GLoVE vectors from file: %s" % glove_path)
    vocab_size = int(400000)  # this is the vocab size of the corpus we've downloaded

    emb_matrix = np.zeros((vocab_size + len(_START_VOCAB), glove_dim))
    word2id = {}
    id2word = {}

    random_init = True
    # randomly initialize the special tokens
    if random_init:
        emb_matrix[:len(_START_VOCAB), :] = np.random.randn(len(_START_VOCAB), glove_dim)

    # put start tokens in the dictionaries
    idx = 0
    for word in _START_VOCAB:
        word2id[word] = idx
        id2word[idx] = word
        idx += 1

    # go through glove vecs
    with open(glove_path, 'r') as fh:
        for line in fh:
            line = line.lstrip().rstrip().split(" ")
            word = line[0]
            vector = list(map(float, line[1:]))
            if glove_dim != len(vector):
                raise Exception(
                    "You set --glove_path=%s but --embedding_size=%i. If you set --glove_path yourself then "
                    "make sure that --embedding_size matches!" % (glove_path, glove_dim))
            emb_matrix[idx, :] = vector
            word2id[word] = idx
            id2word[idx] = word
            idx += 1

    final_vocab_size = vocab_size + len(_START_VOCAB)
    assert len(word2id) == final_vocab_size
    assert len(id2word) == final_vocab_size
    assert idx == final_vocab_size

    return emb_matrix, word2id, id2word


def plotting(visualizeIdx, visualizeVecs_glove, visualizeWords, save_name):
    print("Start plotting ... ")
    visualizeVecs = visualizeVecs_glove
    temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
    covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
    U, S, V = np.linalg.svd(covariance)
    coord = temp.dot(U[:, 0:2])

    # l = len(visualizeWords)
    l = [1, 1, 1, 1, 1]

    print(visualizeVecs.shape)
    color = ['k', 'blue', "green", 'yellow', 'red']

    offset = 0
    cnt = 0
    for curr_l in l:
        for i in range(curr_l):
            print(i+offset)
            plt.text(coord[i+offset, 0], coord[i+offset, 1], visualizeWords[i+offset],
                     bbox=dict(facecolor=color[cnt], alpha=0.1))
        offset += curr_l
        cnt += 1

    plt.xlim((np.min(coord[:, 0]), np.max(coord[:, 0])))
    plt.ylim((np.min(coord[:, 1]), np.max(coord[:, 1])))

    plt.savefig(save_name)


def main():
    # generate random indices
    ids = []
    # for x in range(25):
    #     ids.append(random.randint(0,400001))
    # Load embedding matrix and vocab mappings
    print("Loading glove ...")
    emb_matrix, word2id, id2word = get_glove('./glove.6B.300d.txt', 300)

    print("Look up word indices.")

    words0 = ["coffee"]
    words1 = ["woman"]
    words2 = ["man"]
    words3 = ["king"]
    words4 = ["queen"]
    # words1 = ["espresso", "coffeepot", "coffee", "mug", "eggnog", "maker"]
    # words2 = ["toaster", "french", "loaf", "plate", "potpie"]
    # words3 = ["library", "desk", "grocery", "restaurant", "prison"]
    # words4 = ["menu", "website", "cornet", "barber"]

    ids = []
    for x in words0:
        ids.append(word2id[x])
    # print(ids)
    glove_vectors_0 = emb_matrix[ids]

    ids = []
    for x in words1:
        ids.append(word2id[x])
    # print(ids)
    glove_vectors_1 = emb_matrix[ids]

    ids = []
    for x in words2:
        ids.append(word2id[x])
    # print(ids)
    glove_vectors_2 = emb_matrix[ids]

    ids = []
    for x in words3:
        ids.append(word2id[x])
    # print(ids)
    glove_vectors_3 = emb_matrix[ids]

    ids = []
    for x in words4:
        ids.append(word2id[x])
    # print(ids)
    glove_vectors_4 = emb_matrix[ids]

    glove_vectors = np.concatenate((glove_vectors_0, glove_vectors_1, glove_vectors_2, glove_vectors_3, glove_vectors_4))
    words = words0 + words1 + words2 + words3 + words4

    plotting(ids, glove_vectors, words, "test.png")


if __name__ == "__main__":
    main()
