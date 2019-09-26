import json
import sys

with  open(sys.argv[1], "r") as log_file:
  lines = log_file.readlines()

run_name = lines[2].strip().split(": ")[1].strip(",'")

avg_train_loss = json.loads(lines[-3].strip().split(":")[1])
avg_val_loss = json.loads(lines[-2].strip().split(":")[1])
avg_val_r = json.loads(lines[-1].strip().split(":")[1])

print(",".join(["epoch", 
                  "avg_train_loss",
                  "avg_val_loss",
                  "avg_val_corr",
                  "run"]
                  ))
  
  
for epoch, _ in enumerate(avg_train_loss):
  print(",".join([str(epoch), 
                  str(avg_train_loss[epoch]),
                  str(avg_val_loss[epoch]),
                  str(avg_val_r[epoch]),
                  run_name]))

