import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from utils.data_utils import TokenDataset,pad_collate_fn,RewardModelTokenDataset
import tiktoken
from models.network import GPT,GPT_Reward
import argparse
import yaml
import json
import random
import requests
import shutil
import os
from config.config_classes import load_config
import datetime

##############################################################################
################### global

parser = argparse.ArgumentParser(description = "Reward model")
parser.add_argument('--config',type=str, default='config/config_default.yaml',help='path to config yaml')
parser.add_argument('--train',help='Train the reward mode',action= 'store_true')
args = parser.parse_args()

##############################
##### aditional configurations


##############################
################### global

train_loss_list = []
val_loss_list   = []
steps           = []

###############################################################################

@torch.no_grad()
def generate_multiple_answers(input_sentence,cfg,EOT=True):

    state_dict = torch.load(cfg.rwm.prompting_model_path, map_location=cfg.hypp.device, weights_only=True)
    model = GPT(n_vocab     =  cfg.tknz.n_vocab,
                embd_dim    =  cfg.hypp.embd_dim,
                window_size =  cfg.hypp.window_size,
                n_head      =  cfg.hypp.n_head,
                n_layer     =  cfg.hypp.n_layer,
                dropout     =  cfg.hypp.dropout,
                use_dyt     =  cfg.hypp.use_dyt,
                dyt_alpha   =  cfg.hypp.dyt_alpha
                ).to(cfg.hypp.device)
    model.load_state_dict(state_dict,strict=False)
    model.eval()

    output = []
    for _ in range(cfg.rwm.K):
        output_sentence = model.generate(input_sentence,cfg.rwm.max_prompted_tokens,EOT=True).squeeze(0)
        output.append(output_sentence) #[len(input_sentence[0]):])
    output = torch.nn.utils.rnn.pad_sequence(output,batch_first=True,padding_value=0)
    return output

def compute_loss(reward_model,x,loss,cfg):

    x = x.to(cfg.hypp.device)
    x = generate_multiple_answers(x,cfg)
    logits = reward_model(x[:,-cfg.hypp.window_size:])

    decoded_text = []
    for k in range(cfg.rwm.K):
        decoded_text.append(cfg.tknz.tokenizer.decode(x[k].tolist()))

    for i in range(cfg.rwm.K):
        for j in range(i+1,cfg.rwm.K):
            cfg.rwm.payload['prompt']=decoded_text[i]+'\n'+decoded_text[j]+'\n'+cfg.rwm.prompt
            # cfg.rwm.payload['prompt']=cfg.rwm.payload['prompt']
            response     = requests.post(cfg.rwm.url, json=cfg.rwm.payload)
            first_better = int(response.json()['response'])
            assert(first_better ==1 or first_better ==-1), "Annatator refuses to give reasonable response"
            loss-=torch.log(torch.sigmoid( first_better*(logits[i]-logits[j])))
    loss=loss/(cfg.rwm.K*(cfg.rwm.K-1)/2)
    return loss

@torch.no_grad()
def evaluate_rm(model,dataloader_train,dataloader_vali,cfg,step,total_params):
    model.eval()

    train_iter   = iter(dataloader_train)
    train_losses = torch.zeros(cfg.rwm.eval_iters)
    for eval_iter in range(cfg.rwm.eval_iters):
        x = next(train_iter)

        loss = torch.zeros([1],dtype=torch.float).to(cfg.hypp.device)
        loss = compute_loss(model,x,loss,cfg)

        train_losses[eval_iter] = loss.item()
    train_loss = train_losses.mean()

    val_iter   = iter(dataloader_vali)
    val_losses = torch.zeros (cfg.rwm.eval_iters)
    for eval_iter in range (cfg.rwm.eval_iters):
        x = next(val_iter)

        loss = torch.zeros([1],dtype=torch.float).to(cfg.hypp.device)
        loss = compute_loss(model,x,loss,cfg)

        val_losses[eval_iter] = loss.item()
    val_loss = val_losses.mean()

    print(f"global step (x{cfg.rwm.eval_itvl}) {step//cfg.rwm.eval_itvl}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
    steps.append(step)

    train_loss_list.append(float(train_loss))
    val_loss_list.append(float(val_loss))
    losses={"n_layer":      cfg.hypp.n_layer,
            "embd_dim":     cfg.hypp.embd_dim,
            "window_size":  cfg.hypp.window_size,
            "batch_size":   cfg.rwm.batch_size,
            "total_params": total_params,
            "steps": steps,
            "train_losses": train_loss_list,
            "val_losses":   val_loss_list}
    with open(cfg.rwm.loss_path,'w') as f:
        json.dump(losses,f,indent=4)
    model.train()

def train_reward_model(cfg):

    dataset_train    = RewardModelTokenDataset(cfg.rwm.train_data,cfg.hypp.window_size)
    dataloader_train = DataLoader(dataset_train, batch_size = 1 ,shuffle=True)# batch is generated later by permutation #,collate_fn=pad_collate_fn)
    dataset_vali     = RewardModelTokenDataset(cfg.rwm.vali_data,cfg.hypp.window_size)
    dataloader_vali  = DataLoader(dataset_vali, batch_size = 1 ,shuffle=True)

    state_dict   = torch.load(cfg.rwm.model_init_path, map_location=cfg.hypp.device, weights_only=True)
    reward_model = GPT_Reward( n_vocab     = cfg.tknz.n_vocab,
                               embd_dim    = cfg.hypp.embd_dim,
                               window_size = cfg.hypp.window_size,
                               n_head      = cfg.hypp.n_head,
                               n_layer     = cfg.hypp.n_layer,
                               dropout     = cfg.hypp.dropout
                              ).to(cfg.hypp.device)
    reward_model.load_state_dict(state_dict,strict=False)

    total_params = sum(p.numel() for p in reward_model.parameters())
    print(f"Number of parameters: {total_params}")
    optimizer   = torch.optim.AdamW(reward_model.parameters(),lr=cfg.rwm.lr)
    global_step = 0
    for epoch in range(cfg.rwm.epochs):
        print("Training Reward Model, epoch: ",epoch,"global_step: ",global_step)

        for x in dataloader_train:
            if global_step % cfg.rwm.eval_itvl == 0:
                # want to check what x takes?
                # print(cfg.tknz.tokenizer.decode(x[0].tolist()))

                evaluate_rm(reward_model,dataloader_train,dataloader_vali,cfg,global_step,total_params)
                torch.save(reward_model.state_dict(),cfg.rwm.model_path)

            loss = torch.zeros([1],dtype=torch.float).to(cfg.hypp.device)
            loss = compute_loss(reward_model,x,loss,cfg)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_step+=1

    print("Training Reward Model, epoch: ",epoch,"global_step: ",global_step)
    evaluate_rm(reward_model,dataloader_train,dataloader_vali,cfg,global_step,total_params)
    torch.save(reward_model.state_dict(),cfg.rwm.model_path)
    print("model saved to ",cfg.rwm.model_path)
    total_params = sum(p.numel() for p in reward_model.parameters())
    print(f"Number of parameters: {total_params}")

if __name__ == '__main__':

    cfg = load_config(args.config)
    os.makedirs(cfg.rwm.folder,exist_ok=True)
    os.makedirs(os.path.dirname(cfg.rwm.model_path),exist_ok=True)
    if args.train:

        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        config_save_path = os.path.join(cfg.rwm.folder, f"config_{timestamp}.yaml.log")
        shutil.copyfile(args.config, config_save_path)

        train_reward_model(cfg)
