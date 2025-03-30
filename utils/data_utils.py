import torch
from torch.utils.data import Dataset,IterableDataset,DataLoader
import json
import random
import tiktoken
import importlib
import yaml

def pad_collate_fn(batch):
    xb, yb = zip(*batch)
    xb = torch.nn.utils.rnn.pad_sequence(xb,batch_first=True,padding_value=0)
    yb = torch.nn.utils.rnn.pad_sequence(yb,batch_first=True,padding_value=0)
    return xb,yb

class StreamingTokenDataset(IterableDataset):
    """" One item means one Text"""
    def __init__(self,data_file,window_size):
        self.data_file   = data_file
        self.window_size = window_size

    def __iter__(self):
        with open(self.data_file,'r') as f:
            for line in f:
                tokens = json.loads(line.strip())
                data   = torch.tensor(tokens,dtype=torch.long)
                if len(data)<2:
                    continue
                if len(data) > self.window_size:
                    ix = random.randint(0, len(data)-self.window_size)
                    xb = data[ix:ix+self.window_size]
                    yb = data[ix+1:ix+self.window_size+1]
                else:
                    xb = data[:-1]
                    yb = data[1:]
                yield xb,yb

class TokenDataset(Dataset):
    """" One item means one Text"""
    def __init__(self,data_file,window_size):
        super().__init__()
        self.data_file   = data_file
        self.window_size = window_size
        with open (self.data_file,'r') as f:
            self.data = json.load(f)
        self.data     = [d for d in self.data if len(d)<=self.window_size+1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        data=torch.tensor(self.data[idx],dtype=torch.long)
        assert len(data) >= 2, "text length smaller than 2"
        xb = data[:-1]
        yb = data[1:]
        return xb,yb

class RewardModelTokenDataset(TokenDataset):
    """" One item means one context + one question for now, TODO: extend for more labled
         data
    """
    def __init__(self,data_file,window_size):
        super().__init__(data_file,window_size)

    def __getitem__(self,idx):
        data = torch.tensor(self.data[idx],dtype=torch.long)
        xb   = data[:self.window_size]
        return xb

class RlhfTokenDataset(TokenDataset):
    """" input tokens for prompting"""
    def __init__(self,data_file,window_size):
        super().__init__(data_file,window_size)

    def __getitem__(self,idx):
        data = torch.tensor(self.data[idx],dtype=torch.long)
        xb   = data[-self.window_size:]
        return xb

class PPODataset(Dataset):
    """" One item means one Text"""
    def __init__(self,window_size,state_arr,action_arr,old_prob_arr,values_arr,advantage_arr,sft_prob_arr):
        super().__init__()
        self.window_size   = window_size
        self.states_init   = state_arr
        self.action_arr    = action_arr
        self.old_prob_arr  = old_prob_arr
        self.values_arr    = values_arr
        self.advantage_arr = advantage_arr
        self.sft_prob_arr  = sft_prob_arr

    def __len__(self):
        return len(self.states_init)

    def __getitem__(self,idx):
        tokens = self.states_init[idx].squeeze(0)
        length = len(tokens)
        state  = torch.zeros([self.window_size],dtype=torch.long)
        mask   = torch.zeros([self.window_size],dtype=torch.bool)
        state[:length] = tokens
        mask[:length] = 1

        if length > self.window_size:
            state[:] = tokens[-self.window_size:]
            mask[:]  = 1
        else:
            state[:length] = tokens
            mask[:length]  = 1
        action    = self.action_arr[idx]
        old_probs = self.old_prob_arr[idx]
        advantage = self.advantage_arr[idx]
        values    = self.values_arr[idx]
        sft_prob  = self.sft_prob_arr[idx]

        return state,mask,action,old_probs, values, advantage, sft_prob


class PPOCollection:
    def __init__(self):

        self.states      = []
        self.next_states = []
        self.probs       = []
        self.values      = []
        self.next_values = []
        self.actions     = []
        self.rewards     = []
        self.dones       = []
        self.advantage   = []
        self.sft_prob    = []

        self.states_collection      = []
        self.next_states_collection = []
        self.probs_collection       = []
        self.values_collection      = []
        self.next_values_collection = []
        self.actions_collection     = []
        self.rewards_collection     = []
        self.dones_collection       = []
        self.advantage_collection   = []
        self.sft_prob_collection    = []

    def clear_trajectory(self):
        self.states      = []
        self.next_states = []
        self.probs       = []
        self.values      = []
        self.next_values = []
        self.actions     = []
        self.rewards     = []
        self.dones       = []
        self.advantage   = []
        self.sft_prob    = []

    def clear_collection(self):
        self.states_collection      = []
        self.next_states_collection = []
        self.probs_collection       = []
        self.values_collection      = []
        self.next_values_collection = []
        self.actions_collection     = []
        self.rewards_collection     = []
        self.dones_collection       = []
        self.advantage_collection   = []
        self.sft_prob_collection    = []

        self.clear_trajectory()

    def collect_trajectory(self, state,next_state, action, probs, value,next_value, reward, done,sft_prob):
        self.states.append(state)
        self.next_states.append(next_state)
        self.actions.append(action)
        self.probs.append(probs)
        self.values.append(value)
        self.next_values.append(next_value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.sft_prob.append(sft_prob)


    def trajectory_add(self):
        if self.states == []:
            self.clear_trajectory()
            return len(self.states_collection)

        self.states_collection    += self.states#.append(self.states) #+=self.states.tolist()
        self.probs_collection     += self.probs
        self.values_collection    += self.values
        self.actions_collection   += self.actions
        self.rewards_collection   += self.rewards
        self.dones_collection     += self.dones
        self.advantage_collection += self.advantage
        self.sft_prob_collection  += self.sft_prob

        self.clear_trajectory()

        return len(self.states_collection)

    def collected_data(self):
        return (self.states_collection,
                self.actions_collection,
                self.probs_collection,
                self.values_collection,
                self.rewards_collection,
                self.dones_collection,
                self.advantage_collection,
                self.sft_prob_collection)

    def process_trajectory(self, gamma_rlhf, gae_lambda_rlhf=1.):
        self.advantage = [0] * len(self.rewards)
        gae = 0
        for t in reversed(range(len(self.rewards) )):
            delta = self.rewards[t] + gamma_rlhf * self.next_values[t] - self.values[t]
            gae   = delta + gamma_rlhf * gae_lambda_rlhf * gae
            self.advantage[t] = gae


    def process_trajectory_origin(self,gamma_rlhf,gae_lambda_rlhf): # no GAE, no Rollout
        for t in range(len(self.rewards)):
            a_t = self.rewards[t] + gamma_rlhf * self.next_values[t] - self.values[t]
            self.advantage.append(a_t)

    def process_trajectory_roll_out(self,gamma_rlhf,gae_lambda_rlhf): #not called
        roll_out_steps=4
        for t in range(len(self.rewards)):
            roll_out_steps_onsite=min(roll_out_steps,len(self.rewards)-t)
            return_value =self.next_values[t+roll_out_steps_onsite-1]
            for step in range(roll_out_steps_onsite)[::-1]:
                return_value = self.rewards[t+step]+gamma_rlhf*return_value

            a_t = return_value-self.values[t]
            self.advantage.append(a_t)
