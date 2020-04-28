import os
from tqdm import tqdm
from transformers import BertTokenizer


bert_model = 'bert-large-uncased'
bert_tokenizer = BertTokenizer.from_pretrained(bert_model)


def bert_some_pos(s, bert_tokenizer):
    s = "[CLS] " + s + " [SEP]"
    tokenized_text = bert_tokenizer.tokenize(s)
    return tokenized_text.index("some")


load_db = "./datasets/qualitative.txt"
with open(load_db, "r") as qual_file:
    sentences = [x.strip() for x in qual_file.readlines()]
output = []
for s in tqdm(sentences, total=len(sentences)):
    output.append(bert_some_pos(s, bert_tokenizer))
print(output)
    
