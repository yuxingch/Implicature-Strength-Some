# import os
# import re
# import numpy as np
# from tqdm import tqdm
# from transformers import BertTokenizer


# bert_model = 'bert-large-uncased'
# bert_tokenizer = BertTokenizer.from_pretrained(bert_model)


# def bert_of_pos(s, bert_tokenizer):
#     s = "[CLS] " + s + " [SEP]"
#     tokenized_text = bert_tokenizer.tokenize(s)
#     ids = [i for i, d in enumerate(tokenized_text) if d == 'of']
#     return ids, tokenized_text

# attn = np.load("./best_model/qual_attn_epoch190.npy")

# with open("./all_sent_qual.txt", "r") as f:
#     sentences = [x.strip() for x in f.readlines()]

# with open("./all_sent_of_status_1.csv", "w") as nf:
#     line = "weight,occur_after_some"
#     nf.write(line+"\n")
#     it = 0
#     count = 0
#     for s in tqdm(sentences, total=len(sentences)):
#         curr_ids, tokenized = bert_of_pos(s, bert_tokenizer)
#         # print(curr_ids, tokenized)

#         if len(curr_ids) > 1:
#             count += 1
#         #     for look_up_index in curr_ids:
#         #         # print(look_up_index)
#         #         if look_up_index > 1 and look_up_index < 29 and tokenized[look_up_index-1] == "some":
#         #             line = format(attn[it, look_up_index, 0]) + ",yes"
#         #         elif look_up_index > 0 and look_up_index < 29:
#         #             line = format(attn[it, look_up_index, 0]) + ",no"
#         #         nf.write(line+"\n")
#         # it += 1
# print(count)
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer


bert_model = 'bert-large-uncased'
bert_tokenizer = BertTokenizer.from_pretrained(bert_model)


def bert_of_pos(s, bert_tokenizer):
    s = "[CLS] " + s + " [SEP]"
    tokenized_text = bert_tokenizer.tokenize(s)
    ids = [i for i, d in enumerate(tokenized_text) if d == 'of' and i < 30]
    return ids, tokenized_text

# attn = np.load("./best_model/qual_attn_epoch190.npy")
attn = np.load("./attention_weights/all_weights.npy")

filename = "./some_database.csv"
colnames = ["Item", "workerid", "Rating", "Partitive", "StrengthSome", "Mention", "Subjecthood", "Modification", "Sentence", "SentenceLength", "Trial"]
input_df = pd.read_csv(filename, sep='\t', names=colnames)
# subject_df = input_df[['Item','Subjecthood']].drop_duplicates().groupby('Item')['Subjecthood'].apply(list).to_dict()
sentence_df = input_df[['Item','Sentence']].drop_duplicates().groupby('Item')['Sentence'].apply(list).to_dict()

id_order_fn = "./attention_weights/preds.csv"
colnames = ["Item_ID", "original_mean", "predicted"]
id_fn = pd.read_csv(id_order_fn, sep='\t', names=colnames)
all_ids = id_fn.Item_ID.tolist()[1:]

# with open("./all_sent_qual.txt", "r") as f:
#     sentences = [x.strip() for x in f.readlines()]

it = 0
for i in tqdm(range(attn.shape[0]), total=attn.shape[0]):
    curr_id = all_ids[i]
    s = sentence_df[curr_id][0]
    curr_ids, tokenized = bert_of_pos(s, bert_tokenizer)
    if len(curr_ids) > 1:
        it += 1
print(it)

# with open("./attention_weights/of_status.csv", "w") as nf:
#     line = "weight,occur_after_some"
#     nf.write(line+"\n")
#     it = 0
#     for i in tqdm(range(attn.shape[0]), total=attn.shape[0]):
#         curr_id = all_ids[i]
#         s = sentence_df[curr_id][0]
#         curr_ids, tokenized = bert_of_pos(s, bert_tokenizer)
#         # print(curr_ids, tokenized)
#         # if len(curr_ids) > 1:
#         if len(curr_ids) > 0:
#             values = [attn[it, v, 0] for v in curr_ids]
#             # values_sum = sum(values)
#             # values_norm = [v/values_sum for v in values]
#             curr_id_it = 0
#             for look_up_index in curr_ids:
#                 if tokenized[look_up_index-1] == "some":
#                     # line = format(attn[it, look_up_index, 0]) + ",yes"
#                     # line = format(values_norm[curr_id_it]) + ",yes"
#                     line = format(values[curr_id_it]) + ",yes"
#                 else:
#                     # line = format(attn[it, look_up_index, 0]) + ",no"
#                     # line = format(values_norm[curr_id_it]) + ",no"
#                     line = format(values[curr_id_it]) + ",no"
#                 curr_id_it += 1
#                 nf.write(line+"\n")
#         it += 1
