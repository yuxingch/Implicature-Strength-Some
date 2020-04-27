import os
import re
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer


bert_model = 'bert-large-uncased'
bert_tokenizer = BertTokenizer.from_pretrained(bert_model)


candidates = ["these", "those", "this", "that", "them", "it", "him", "her", "its"]


def bert_of_pos(s, bert_tokenizer):
    s = "[CLS] " + s + " [SEP]"
    tokenized_text = bert_tokenizer.tokenize(s)
    ids = [i for i, d in enumerate(tokenized_text) if d == 'of' and i > 1 and i < 29]
    return ids, tokenized_text


# for articial sentences
def np_to_pronoun(filename):
    return


# for natural sentences
def pronoun_to_np(filename):
    return


def filter_of(filename):
    with open(filename + ".txt", "r") as f:
        sentences = [x.strip() for x in f.readlines()]

    with open(filename + "_partitive_compare.txt", "w") as nf:
        for s in tqdm(sentences, total=len(sentences)):
            curr_ids, tokenized = bert_of_pos(s, bert_tokenizer)
            flag = False
            if len(curr_ids) > 0:
                for look_up_index in curr_ids:
                    if tokenized[look_up_index-1] == "some":
                        if tokenized[look_up_index+1] in candidates:
                            flag = True
                if flag:
                    nf.write(s+"\n")


if __name__ == "__main__":
    filter_of("./qualitative_results")
