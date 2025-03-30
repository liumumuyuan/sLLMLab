from datasets import load_dataset
import tiktoken
import json
import os
from config.config_classes import load_config
import argparse

parser  = argparse.ArgumentParser(description = "Data preparation for reward model")
parser.add_argument('--config',type=str, default='config/config_default.yaml',help='path to config yaml')
args    = parser.parse_args()

cfg            = load_config(args.config)
data_folder    = cfg.dprp.sft_data_folder
sft_train_data = cfg.dprp.sft_train_data
sft_vali_data  = cfg.dprp.sft_vali_data
tokenizer      = cfg.tknz.tokenizer

dataset        = load_dataset('squad')
print('data:',dataset)
print('dataset["train"][0]:',dataset['train'][0])

os.makedirs(data_folder, exist_ok=True)

with open(os.path.join(data_folder,sft_train_data),'w') as f:
    tokenized = []
    for data in dataset['train']:
        context_Q_A_eot         = data['context']+' '+data['question']+' '+ data['answers']['text'][0]
        context_Q_A_eot_encoded = tokenizer.encode(context_Q_A_eot)
        context_Q_A_eot_encoded.append(tokenizer.eos_token_id) #50256 for gpt2
        tokenized.append(context_Q_A_eot_encoded)
    json.dump(tokenized,f)

with open(os.path.join(data_folder,sft_vali_data),'w') as f:
    tokenized = []
    for data in dataset['validation']:
        context_Q_A_eot         = data['context']+' '+data['question']+' '+ data['answers']['text'][0]
        context_Q_A_eot_encoded = tokenizer.encode(context_Q_A_eot)
        context_Q_A_eot_encoded.append(tokenizer.eos_token_id)
        tokenized.append(context_Q_A_eot_encoded)
    json.dump(tokenized,f)

print("data saved to "+ data_folder)




