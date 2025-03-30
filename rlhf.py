import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from utils.data_utils import TokenDataset,pad_collate_fn,RlhfTokenDataset,PPOCollection,PPODataset
import tiktoken
from models.network import GPT,GPT_Reward,Actor_Critic,SFT_RLHF
import argparse
import yaml
import json
import random
import requests
import shutil
import os
from torch.distributions.categorical import Categorical
from config.config_classes import load_config
import datetime
##############################################################################
parser = argparse.ArgumentParser(description = "Reinforcement Learning with Human Feedback")
parser.add_argument('--config',type=str, default='config/config_default.yaml',help='path to config yaml')
parser.add_argument('--train',help='Training mode',action= 'store_true')
args   = parser.parse_args()

##############################
##### aditional configurations


##############################
################### global

steps          = []
scores         = []
best_scores    = []
average_scores = []

################################################################3

class RLHF():
    def __init__(self,cfg,actor_critic_model,reward_model,sft_model,ppo_collection):
        self.gamma               = cfg.rlhf.gamma
        self.gae_lambda          = cfg.rlhf.gae_lambda
        self.policy_clip         = cfg.rlhf.policy_clip
        self.epochs_experience   = cfg.rlhf.epochs_experience
        self.inner_batch_size    = cfg.rlhf.inner_batch_size

        self.collection          = ppo_collection
        self.collection_len      = 0
        self.reward_model        = reward_model
        self.sft_model           = sft_model
        self.actor_critic_model  = actor_critic_model

        self.optimizer = torch.optim.AdamW(self.actor_critic_model.parameters(),lr=cfg.rlhf.lr)

    def collect_trajectory( self,           state,      next_state,
                                            action,     probs,
                                            value,      next_value,
                                            reward,     done,
                                            sft_prob):
        self.collection.collect_trajectory( state,      next_state,
                                            action,     probs,
                                            value,      next_value,
                                            reward,     done,
                                            sft_prob)

    def process_trajectory(self):
        self.collection.process_trajectory(self.gamma,self.gae_lambda)

    def trajectory_add(self):
        self.collection_len=self.collection.trajectory_add()

    @torch.no_grad()
    def step(self,s):
        self.actor_critic_model.eval()
        return self.actor_critic_model.step(s)

    @torch.no_grad()
    def reward(self,s):
        self.reward_model.eval()
        return self.reward_model(s)

    @torch.no_grad()
    def sft_step(self,s):
        self.sft_model.eval()
        return self.sft_model.step(s)

    def action_value(self,s,attention_mask=None):
        return self.actor_critic_model.actor_critic(s,attention_mask)

    def learn(self):

        self.actor_critic_model.train()
        state_arr, action_arr, old_prob_arr, values_arr, _, _,advantage_arr,sft_prob_arr = self.collection.collected_data()

        ppo_dataset=PPODataset( cfg.hypp.window_size,
                                state_arr,
                                action_arr,
                                old_prob_arr,
                                values_arr,
                                advantage_arr,
                                sft_prob_arr)

        ppo_dataloader_train = DataLoader(ppo_dataset, batch_size = self.inner_batch_size ,shuffle=True)

        for _ in range(self.epochs_experience):
            for state,attention_mask,actions,old_probs,values,advantage,sft_prob in ppo_dataloader_train:
                advantage         = advantage.detach()
                values            = values.detach()
                old_probs         = old_probs.detach()
                sft_prob          = sft_prob.detach()

                state=state.to(cfg.hypp.device)
                attention_mask    = attention_mask.unsqueeze(-1)
                attention_mask    = attention_mask.to(cfg.hypp.device)

                dist,critic_value = self.action_value(state,attention_mask)

                returns           = advantage + values
                critic_loss       = (returns - critic_value)**2
                critic_loss       = critic_loss.mean()

                new_probs         = dist.log_prob(actions.squeeze(-1))
                new_probs         = new_probs.unsqueeze(-1)
                prob_ratio        = new_probs.exp() / old_probs.exp()
                weighted_probs    = advantage * prob_ratio
                clipped_probs     = torch.clamp(prob_ratio, 1 - self.policy_clip,
                                                 1 + self.policy_clip) * advantage

                actor_loss        = -torch.min(weighted_probs, clipped_probs).mean()

                sft_prob_ratio    = torch.log(new_probs.exp() / sft_prob.exp())
                sft_loss          = sft_prob_ratio.mean()

                total_loss        = actor_loss + 0.5 * critic_loss #TODO, the weighting coefficients should be given in config file
                total_loss        = total_loss - 0.1 * dist.entropy().mean()
                total_loss        = total_loss + 0.1 * sft_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

        self.collection.clear_collection()

def evaluate_and_save_rlhf(rlhf,step,score,average_score,best_score,cfg):

    if average_score > best_score:
        best_score      = average_score
        full_state_dict = rlhf.actor_critic_model.state_dict()
        gpt_state_dict  = {k: v for k, v in full_state_dict.items() if not k.startswith("Linear_critic")}
        torch.save(gpt_state_dict, cfg.rlhf.result_model_path)
        print(f'model saved to {cfg.rlhf.result_model_path}')
    # print(step,score.item(),best_score.item(),average_score.item())
    steps.append(step)
    scores.append(score.item())
    best_scores.append(best_score.item())
    average_scores.append(average_score.item())
    score_logs={  "steps":    steps,
                 "scores":    scores,
            "best scores":    best_scores,
            "average_scores": average_scores}
    with open(cfg.rlhf.scores_path,'w') as f:
        json.dump(score_logs,f,indent=4)

    print(f"global step (x{cfg.rlhf.eval_itvl}) {step//cfg.rlhf.eval_itvl}: score {score.item():.4f}, best_score {best_score.item():.4f}, average_score {average_score.item(): 4f}")

    return best_score

def train():

    dataset_train    = RlhfTokenDataset(cfg.rlhf.train_data,cfg.hypp.window_size)
    dataloader_train = DataLoader(dataset_train, batch_size = 1 ,shuffle=True)
    #batch_size =1: one prompt unfolds one episode
    ppo_collection   = PPOCollection()

    actor_critic_model = Actor_Critic(n_vocab     = cfg.tknz.n_vocab,
                                      embd_dim    = cfg.hypp.embd_dim,
                                      window_size = cfg.hypp.window_size,
                                      n_head      = cfg.hypp.n_head,
                                      n_layer     = cfg.hypp.n_layer,
                                      dropout     = cfg.hypp.dropout
                                      ).to(cfg.hypp.device)
    state_dict = torch.load(cfg.rlhf.actor_critic_model_init_path,
                            map_location=cfg.hypp.device,
                            weights_only=True)
    actor_critic_model.load_state_dict(state_dict,strict=False)

    reward_model = GPT_Reward(n_vocab     = cfg.tknz.n_vocab,
                              embd_dim    = cfg.hypp.embd_dim,
                              window_size = cfg.hypp.window_size,
                              n_head      = cfg.hypp.n_head,
                              n_layer     = cfg.hypp.n_layer,
                              dropout     = cfg.hypp.dropout
                              ).to(cfg.hypp.device)
    state_dict = torch.load(cfg.rlhf.reward_model_path,
                            map_location=cfg.hypp.device,
                            weights_only=True)
    reward_model.load_state_dict(state_dict)

    sft_model = SFT_RLHF(n_vocab     = cfg.tknz.n_vocab,
                         embd_dim    = cfg.hypp.embd_dim,
                         window_size = cfg.hypp.window_size,
                         n_head      = cfg.hypp.n_head,
                         n_layer     = cfg.hypp.n_layer,
                         dropout     = cfg.hypp.dropout
                         ).to(cfg.hypp.device)
    state_dict = torch.load(cfg.rlhf.sft_model_path,
                            map_location=cfg.hypp.device,
                            weights_only=True)
    sft_model.load_state_dict(state_dict)

    rlhf = RLHF(cfg,actor_critic_model,reward_model,sft_model, ppo_collection)

    total_params_sft = sum(p.numel() for p in rlhf.sft_model.parameters())
    print(f"Number of parameters in sft_model: {total_params_sft}")
    total_params_reward = sum(p.numel() for p in rlhf.reward_model.parameters())
    print(f"Number of parameters in reward model: {total_params_reward}")
    total_params_actor_critic = sum(p.numel() for p in rlhf.actor_critic_model.parameters())
    print(f"Number of parameters in actor_critic_model: {total_params_actor_critic}")

    # TODO: temperature, top_k, top_p

    global_step = 0
    score_total=0
    best_score=-1e9
    for epoch in range(cfg.rlhf.epochs): # epochs for in prompting dataset
        print("RLHF, epoch:",epoch,"global_step:",global_step)
        for x in dataloader_train:# sample one prompt, one episode

            s     = x.to(cfg.hypp.device)
            done  = False
            score = 0
            n_steps_in_episode =0
            while not done:
                action, prob, s_next, done = rlhf.step(s)
                reward                     = rlhf.reward(s_next)
                sft_prob                   = rlhf.sft_step(s)
                _,value                    = rlhf.action_value(s)
                _,next_value               = rlhf.action_value(s_next)
                score += reward

                value=value.squeeze(0)
                next_value=next_value.squeeze(0)
                reward=reward.squeeze(0)

                rlhf.collect_trajectory(s,s_next,action,prob,value,next_value,reward,done,sft_prob)
                s = s_next
                n_steps_in_episode+=1
                if n_steps_in_episode%cfg.rlhf.N==0:
                    rlhf.process_trajectory()
                    rlhf.trajectory_add()

                    if  rlhf.collection_len >= cfg.rlhf.minsize_experience_buffer:
                        rlhf.learn()

                if n_steps_in_episode==cfg.rlhf.max_len_episode:
                    break

            rlhf.process_trajectory()
            rlhf.trajectory_add()

            if rlhf.collection_len >= cfg.rlhf.minsize_experience_buffer:
                rlhf.learn()

            global_step += 1
            score_total += score
            if global_step%cfg.rlhf.eval_itvl == 0:
                average_score = score_total/cfg.rlhf.eval_itvl
                best_score    = evaluate_and_save_rlhf(rlhf,global_step, score, average_score,best_score,cfg)
                score_total   = 0

    print("RLHF, epoch: ",epoch,"global_step: ",global_step)
    print(f"Number of parameters in sft_model: {total_params_sft}")
    print(f"Number of parameters in reward model: {total_params_reward}")
    print(f"Number of parameters in actor_critic_model: {total_params_actor_critic}")

###############################################################################
if __name__ == '__main__':
    cfg = load_config(args.config)
    os.makedirs(cfg.rlhf.folder,exist_ok=True)
    os.makedirs(os.path.dirname(cfg.rlhf.result_model_path),exist_ok=True)
    if args.train:

        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        config_save_path = os.path.join(cfg.rlhf.folder, f"config_{timestamp}.yaml.log")
        shutil.copyfile(args.config, config_save_path)

        train()
