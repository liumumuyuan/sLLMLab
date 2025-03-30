from datasets import load_dataset
import tiktoken
import json
import os
from config.config_classes import load_config
import argparse

parser = argparse.ArgumentParser(description = "Data preparation for pretraining")
parser.add_argument('--config',type=str, default='config/config_default.yaml',help='path to config yaml')
args = parser.parse_args()

cfg         = load_config(args.config)
tokenizer   = cfg.tknz.tokenizer
data_folder = cfg.dprp.pretraining_data_folder
train_data  = cfg.dprp.pretraining_train_data
vali_data   = cfg.dprp.pretraining_vali_data
os.makedirs(data_folder, exist_ok=True)

def write_tokenized_json(dataset, output_file):
    with open(output_file, 'w') as f:
        for data in dataset:
            tokenized_data = tokenizer.encode(data['text'])
            f.write(json.dumps(tokenized_data)+'\n')

train_dataset = load_dataset("allenai/c4", "realnewslike", split="train")#,streaming=True)
write_tokenized_json(train_dataset,os.path.join( data_folder,train_data))

validation_dataset = load_dataset("allenai/c4", "realnewslike", split="validation")#,streaming=True)
write_tokenized_json(validation_dataset, os.path.join(data_folder,vali_data))
print("data saved to "+ data_folder)
