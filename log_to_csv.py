import json
import sys

with  open(sys.argv[1], "r") as log_file:
  lines = log_file.readlines()

json_config_lines = []
json_lines = False
for line in lines:
  if line.strip().startswith("{"):
    json_lines = True
  
  if json_lines:
    json_config_lines.append(line.strip().replace("'", '"').replace("True", "true").replace("False", "false"))
  
  if line.strip().endswith("}"):
    break

json_config = json.loads(" ".join(json_config_lines))    

run_name = json_config["CONFIG_NAME"]
lstm_attn = json_config["LSTM"]["ATTN"]
lstm_bidirectional = json_config["LSTM"]["BIDIRECTION"]
lstm_hiddensize = json_config["LSTM"]["HIDDEN_DIM"]
lstm_dropout = json_config["LSTM"]["DROP_PROB"]
lstm_layers = json_config["LSTM"]["LAYERS"]
elmo_layer = json_config["ELMO_LAYER"] if ("ELMO_LAYER" in json_config and json_config["IS_ELMO"]) else -1
bert_layer = json_config["BERT_LAYER"] if ("BERT_LAYER" in json_config and json_config["IS_BERT"]) else -1
bert_large = json_config["BERT_LARGE"] if "BERT_LARGE" in json_config else False
embedding = "elmo" if json_config["IS_ELMO"] else (
            "bert_large" if json_config["IS_BERT"] and "BERT_LARGE" in json_config and json_config["BERT_LARGE"] else (
            "bert" if json_config["IS_BERT"] else "glove"))



avg_train_loss = json.loads(lines[-3].strip().split(":")[1])
avg_val_loss = json.loads(lines[-2].strip().split(":")[1])
avg_val_r = json.loads(lines[-1].strip().split(":")[1])

print(",".join(["epoch", 
                  "avg_train_loss",
                  "avg_val_loss",
                  "avg_val_corr",
                  "run",
                  "lstm_attn",
                  "lstm_bidirectional",
                  "lstm_hiddensize",
                  "lstm_dropout",
                  "lstm_layers",
                  "elmo_layer",
                  "bert_layer",
                  "bert_large",
                  "embedding"]
                  ))
  
  
for epoch, _ in enumerate(avg_train_loss):
  print(",".join([str(s) for s in [epoch, 
                  avg_train_loss[epoch],
                  avg_val_loss[epoch],
                  avg_val_r[epoch],
                  run_name,
                  lstm_attn,
                  lstm_bidirectional,
                  lstm_hiddensize,
                  lstm_dropout,
                  lstm_layers,
                  elmo_layer,
                  bert_layer,
                  bert_large,
                  embedding]]))

