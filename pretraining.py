import torch
# import torch.nn as nn
from torch.utils.data import DataLoader
from utils.data_utils import StreamingTokenDataset,pad_collate_fn
import tiktoken
from models.network import GPT
import argparse
import yaml
import json
import random
import os
import shutil
from config.config_classes import load_config
import datetime

###################################################################################
parser = argparse.ArgumentParser(description = "Train or Generate")
parser.add_argument('--config',type=str, default='config/config_default.yaml',help='path to config yaml')
parser.add_argument('--train',help='Training mode',action= 'store_true')
parser.add_argument('--generate', type=str, help='Input sentence for generation')
args = parser.parse_args()

##############################
##### aditional configurations


##############################
################### global

train_loss_list = []
val_loss_list = []
steps = []

####################################################################################

@torch.no_grad()
def evaluate_model_prt(model,dataloader_train,dataloader_vali,step,total_params,cfg):

    model.eval()

    train_iter   = iter(dataloader_train)
    train_losses = torch.zeros(cfg.prt.eval_iters)
    for eval_iter in range(cfg.prt.eval_iters):
        x,y = next(train_iter)
        x   = x.to(cfg.hypp.device)
        y   = y.to(cfg.hypp.device)
        logits,loss             = model(x,y)
        train_losses[eval_iter] = loss.item()
    train_loss=train_losses.mean()

    val_iter   =iter(dataloader_vali)
    val_losses = torch.zeros(cfg.prt.eval_iters)
    for eval_iter in range(cfg.prt.eval_iters):
        x,y = next(val_iter)
        x   = x.to(cfg.hypp.device)
        y   = y.to(cfg.hypp.device)
        logits,loss           = model(x,y)
        val_losses[eval_iter] = loss.item()
    val_loss=val_losses.mean()

    print(f"global step (x{cfg.prt.eval_itvl}) {step//cfg.prt.eval_itvl}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
    steps.append(step)

    train_loss_list.append(float(train_loss))
    val_loss_list.append(float(val_loss))
    losses={"n_layer":      cfg.hypp.n_layer,
            "embd_dim":     cfg.hypp.embd_dim,
            "window_size":  cfg.hypp.window_size,
            "batch_size":   cfg.prt.batch_size,
            "total_params": total_params,
            "steps":        steps,
            "train_losses": train_loss_list,
            "val_losses":   val_loss_list}
    with open(cfg.prt.loss_path,'w') as f:
        json.dump(losses,f,indent=4)

    model.train()

def train(cfg):
    dataset_train    = StreamingTokenDataset(cfg.prt.train_data,cfg.hypp.window_size)
    dataloader_train = DataLoader(dataset_train,batch_size = cfg.prt.batch_size, pin_memory=True,num_workers=0,collate_fn=pad_collate_fn)
    dataset_vali     = StreamingTokenDataset(cfg.prt.vali_data,cfg.hypp.window_size)
    dataloader_vali  = DataLoader(dataset_vali,batch_size = cfg.prt.batch_size, pin_memory=True,num_workers=0,collate_fn=pad_collate_fn)

    model= GPT(n_vocab     = cfg.tknz.n_vocab,
               embd_dim    = cfg.hypp.embd_dim,
               window_size = cfg.hypp.window_size,
               n_head      = cfg.hypp.n_head,
               n_layer     = cfg.hypp.n_layer,
               dropout     = cfg.hypp.dropout,
               use_dyt     = cfg.hypp.use_dyt,
               dyt_alpha   = cfg.hypp.dyt_alpha).to(cfg.hypp.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg.prt.lr)

    global_step = 0
    for epoch in range(cfg.prt.epochs):
        print("epoch: ",epoch,f"global step (x{cfg.prt.eval_itvl}): ",global_step//cfg.prt.eval_itvl)

        for xb,yb in dataloader_train:
            if global_step%cfg.prt.eval_itvl == 0:
                evaluate_model_prt(model,dataloader_train,dataloader_vali,global_step,total_params,cfg)
                torch.save(model.state_dict(),cfg.prt.model_path)
            xb = xb.to(cfg.hypp.device)
            yb = yb.to(cfg.hypp.device)
            logits, loss = model(xb,yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_step+=1

    print("epoch: ",epoch,f"global step (x{cfg.prt.eval_itvl}): ",global_step//cfg.prt.eval_itvl)
    torch.save(model.state_dict(),cfg.prt.model_path)
    evaluate_model_prt(model,dataloader_train,dataloader_vali,global_step,total_params,cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")

@torch.no_grad()
def generate(input_sentence,cfg):

    output_sentence = torch.tensor(cfg.tknz.tokenizer.encode(input_sentence),dtype=torch.long).unsqueeze(0).to(cfg.hypp.device)
    state_dict      = torch.load(cfg.prt.model_path, map_location=cfg.hypp.device,weights_only=True)
    model = GPT(n_vocab     =  cfg.tknz.n_vocab,
                embd_dim    =  cfg.hypp.embd_dim,
                window_size =  cfg.hypp.window_size,
                n_head      =  cfg.hypp.n_head,
                n_layer     =  cfg.hypp.n_layer,
                dropout     =  cfg.hypp.dropout,
                use_dyt     =  cfg.hypp.use_dyt,
                dyt_alpha   =  cfg.hypp.dyt_alpha
                ).to(cfg.hypp.device)
    model.load_state_dict(state_dict)
    model.eval()
    output_sentence = model.generate(output_sentence,cfg.hypp.max_new_tokens).squeeze(0).tolist()

    output_sentence = cfg.tknz.tokenizer.decode(output_sentence)
    print("input:", input_sentence)
    print("output:", output_sentence[len(input_sentence):])

if __name__ == '__main__':

    cfg = load_config(args.config)
    os.makedirs(cfg.prt.folder,exist_ok=True)
    os.makedirs(os.path.dirname(cfg.prt.model_path),exist_ok=True)

    if args.train:
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        config_save_path = os.path.join(cfg.prt.folder, f"config_{timestamp}.yaml.log")
        shutil.copyfile(args.config, config_save_path)

    if args.train:
        train(cfg)
    else:
        generate(args.generate,cfg)
