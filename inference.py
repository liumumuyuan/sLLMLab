import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from models.network import GPT
import argparse
import yaml
import json
import random
import requests
import shutil
import os
from torch.distributions.categorical import Categorical
from config.config_classes import load_config

###################################################################################
parser = argparse.ArgumentParser(description = "Inference")
parser.add_argument('inference', type=str, help='Input sentence for Inference')
parser.add_argument('--config',type=str, default='config/config_default.yaml',help='path to config yaml')

args = parser.parse_args()

##############################
##### aditional configurations


##############################
################### global

######################################################################################
@torch.no_grad()
def generate(input_sentence,cfg):
    output_sentence=torch.tensor(cfg.tknz.tokenizer.encode(input_sentence),dtype=torch.long).unsqueeze(0).to(cfg.hypp.device)

    state_dict = torch.load(cfg.inference.model_path, map_location=cfg.hypp.device, weights_only=True)
    model = GPT(n_vocab     = cfg.tknz.n_vocab,
                embd_dim    = cfg.hypp.embd_dim,
                window_size = cfg.hypp.window_size,
                n_head      = cfg.hypp.n_head,
                n_layer     = cfg.hypp.n_layer,
                dropout     = cfg.hypp.dropout,
                use_dyt     = cfg.hypp.use_dyt,
                dyt_alpha   = cfg.hypp.dyt_alpha
                ).to(cfg.hypp.device)
    model.load_state_dict(state_dict)
    model.eval()
    output_sentence=model.generate(output_sentence,cfg.inference.max_new_tokens).squeeze(0).tolist()

    output_sentence=cfg.tknz.tokenizer.decode(output_sentence)
    print(" input:", input_sentence)
    print("output:", output_sentence[len(input_sentence):])

######
###############################################################################

if __name__ == '__main__':

    cfg = load_config(args.config)
    generate(args.inference,cfg)


