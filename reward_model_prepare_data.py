from datasets import load_dataset
import tiktoken
import json
import os
from config.config_classes import load_config
import argparse

parser = argparse.ArgumentParser(description = "Data preparation for reward model")
parser.add_argument('--config',type=str, default='config/config_default.yaml',help='path to config yaml')
args = parser.parse_args()

cfg         = load_config(args.config)
data_folder = cfg.dprp.reward_model_data_folder
train_data  = cfg.dprp.reward_model_train_data
vali_data   = cfg.dprp.reward_model_vali_data
tokenizer   = cfg.tknz.tokenizer

dataset=load_dataset('squad')
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

with open(os.path.join(data_folder,vali_data),'w') as f:
    tokenized = []
    for data in dataset['validation']:
        context_Q         = data['context']+' '+data['question']
        context_Q_encoded = tokenizer.encode(context_Q)
        tokenized.append(context_Q_encoded)
    json.dump(tokenized,f)

print("data saved to "+ data_folder)






