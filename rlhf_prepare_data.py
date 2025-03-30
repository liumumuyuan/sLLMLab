from datasets import load_dataset
import tiktoken
import json
import os
from config.config_classes import load_config
import argparse

parser = argparse.ArgumentParser(description = "Data preparation for RLHF")
parser.add_argument('--config',type=str, default='config/config_default.yaml',help='path to config yaml')
args   = parser.parse_args()

cfg         = load_config(args.config)
data_folder = cfg.dprp.RLHF_data_folder
train_data  = cfg.dprp.RLHF_train_data
vali_data   = cfg.dprp.RLHF_vali_data
tokenizer   = cfg.tknz.tokenizer

# This data preparation is (at present) the same (but can be used for including or switching to other datasets) to what is used for reward model training.
# Similar to rewad model, I only tokenize the context + question part as prompts
# to kick off the RLHF.
# Now it becomes clear: every user input as prompt can be used for RLHF
# And, the request for choosing a favorable answer helps to train the reward model.
# And, collecting user prompts helps distillations since diverse prompts are available.
dataset    = load_dataset('squad')
print('data:',dataset)
print('dataset["train"][0]:',dataset['train'][0])

os.makedirs(data_folder, exist_ok=True)

with open(os.path.join(data_folder,train_data),'w') as f:
    tokenized = []
    for data in dataset['train']:
        context_Q         = data['context']+' '+data['question']
        context_Q_encoded = tokenizer.encode(context_Q)
        tokenized.append(context_Q_encoded)
    json.dump(tokenized,f)
    print("save file done")

with open(os.path.join(data_folder,vali_data),'w') as f:
    tokenized = []
    for data in dataset['validation']:
        context_Q         = data['context']+' '+data['question']
        context_Q_encoded = tokenizer.encode(context_Q)
        tokenized.append(context_Q_encoded)
    json.dump(tokenized,f)

print("data saved to "+ data_folder)


