import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.distributions.categorical import Categorical

###################################################################################
# GPT model is largely based on the implementatio of Andrej Karpathy
# modification: Gelu instead of Relu,
#               Dynamic tanh,
#               attention_mask for padding in RLHF
# TODO: Add support for special tokens at init (pad/eos), i.e. 50256 for eos

###################################################################################
# Inheritances:
# GPT_Reward add additional layer for regressing reward
# SFT_RLHF for regulation term in loss function
# Actor_Critic is the key model for RLHF

##############

class DyT(nn.Module):
    def __init__(self,C,init_alpha=0.2):
        super().__init__()
        self.alpha = Parameter(torch.ones(C)*init_alpha)
        self.gamma = Parameter(torch.ones(C))
        self.beta = Parameter(torch.zeros(C))

    def forward(self,x):
        x = torch.tanh(self.alpha * x)
        return self.gamma*x + self.beta

class Head(nn.Module):
    def __init__(self,window_size, embd_dim,head_size,dropout):
        super().__init__()
        self.key = nn.Linear(embd_dim, head_size,bias=False)
        self.query = nn.Linear(embd_dim,head_size,bias=False)
        self.value = nn.Linear(embd_dim,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(window_size,window_size,dtype=bool)))
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,attention_mask=None):
        B,T,C = x.shape
        #3, 256, 384
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        _,_,dk = k.shape
        w = q@k.transpose(-2,-1) * dk **-0.5
        w = w.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        w = F.softmax(w,dim=-1)
        w = self.dropout(w)
        attention = w@v
        if attention_mask is not None:
            attention = attention.masked_fill(attention_mask==0, 0)
        return attention

class MultiHeadAtten(nn.Module):
    def __init__(self,window_size,embd_dim,n_heads, head_size,dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(window_size,embd_dim,head_size,dropout) for _ in range(n_heads)])
        self.proj = nn.Linear(embd_dim,embd_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,attention_mask=None):
        x = torch.cat([h(x,attention_mask=attention_mask) for h in self.heads],dim=-1)
        out = self.dropout(self.proj(x))
        return out

class FeedForward(nn.Module):
    def __init__(self,embd_dim,dropout):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(embd_dim,4*embd_dim),
                               nn.GELU(), #ReLU
                               nn.Linear(4*embd_dim,embd_dim),
                               nn.Dropout(dropout))
    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self,window_size,embd_dim,n_head,dropout,use_dyt=False,dyt_alpha=0.2):
        super().__init__()
        self.use_dyt=use_dyt
        head_size = embd_dim //n_head
        self.sa = MultiHeadAtten(window_size,embd_dim,n_head,head_size,dropout)
        self.ffwd = FeedForward(embd_dim,dropout)
        if self.use_dyt==True:
            self.dyt1 = DyT(embd_dim,dyt_alpha)
            self.dyt2 = DyT(embd_dim,dyt_alpha)
        else:
            self.ln1 = nn.LayerNorm(embd_dim)
            self.ln2 = nn.LayerNorm(embd_dim)

    def forward(self,x,attention_mask=None):
        if self.use_dyt==True:
            x= x+ self.sa(self.dyt1(x),attention_mask=attention_mask)
            x= x+ self.ffwd(self.dyt2(x))
        else:
            x = x+self.sa(self.ln1(x),attention_mask=attention_mask)
            x = x+self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self,n_vocab,embd_dim,window_size,n_head,n_layer,dropout,use_dyt=False,dyt_alpha=0.2):
        super().__init__()
        self.use_dyt=use_dyt
        self.window_size=window_size
        self.token_embedding = nn.Embedding(n_vocab,embd_dim) #50257 384
        self.position_embedding  = nn.Embedding(window_size,embd_dim)
        self.blocks = nn.ModuleList([Block(window_size,embd_dim,n_head,dropout,use_dyt,dyt_alpha) for _ in range(n_layer)])

        self.lm_head = nn.Linear(embd_dim, n_vocab) #nn.Embedding(..., padding_idx=0)
        self.lm_head.weight = self.token_embedding.weight #weight sharing
        if self.use_dyt==True:
            self.dyt = DyT(embd_dim,dyt_alpha)
        else:
            self.ln_f = nn.LayerNorm(embd_dim)

    def forward(self,x,targets=None,attention_mask=None):

        B,T= x.shape
        #torch.Size([3, 256])
        tok_emb = self.token_embedding(x)
        #torch.Size([3, 256, 384])
        pos_emb = self.position_embedding(torch.arange(T,device=tok_emb.device))
        #torch.Size([256, 384])
        x=tok_emb+pos_emb
        # torch.Size([3, 256, 384])

        for block in self.blocks:
            x=block(x,attention_mask=attention_mask)
        if self.use_dyt==True:
            x=self.dyt(x)
        else:
            x=self.ln_f(x)
        # torch.Size([3, 256, 384])
        logits = self.lm_head(x)

        if targets is None:
            return logits

        else:
            B,T,C=logits.shape
            logits=logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)

        return logits,loss

    def generate(self,x,max_new_tokens=1000,EOT=False):

        for _ in range(max_new_tokens):
            x_cond = x[:,-self.window_size:]
            logits = self(x_cond)
            logits=logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            x_next = torch.multinomial(probs,num_samples=1)
            x = torch.cat((x,x_next),dim=1)
            if EOT==True:
                if x_next==50256:
                    return x
        return x

################

class GPT_Reward(GPT):
    def __init__(self,n_vocab,embd_dim,window_size,n_head,n_layer,dropout):
        super().__init__(n_vocab,embd_dim,window_size,n_head,n_layer,dropout,use_dyt=False,dyt_alpha=0.2)
        self.reward = nn.Linear(embd_dim,1)

    def forward(self,x):
        B,T= x.shape
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(torch.arange(T,device=tok_emb.device))
        x=tok_emb+pos_emb

        for block in self.blocks:
            x=block(x)

        if self.use_dyt==True:
            x=self.dyt(x)
        else:
            x=self.ln_f(x)
        return self.reward(x[:,-1,:]) # use the last token

###################

class SFT_RLHF(GPT):
    def __init__(self,n_vocab,embd_dim,window_size,n_head,n_layer,dropout):
        super().__init__(n_vocab,embd_dim,window_size,n_head,n_layer,dropout,use_dyt=False,dyt_alpha=0.2)

    def step(self,s):
        logits = self(s[:,-self.window_size:])

        logits=logits[:,-1,:]
        dist = F.softmax(logits,dim=-1)
        dist = Categorical(dist)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return log_prob

###################

class Actor_Critic(GPT):
    def __init__(self,n_vocab,embd_dim,window_size,n_head,n_layer,dropout,use_dyt=False,dyt_alpha=0.2):
        super().__init__(n_vocab,embd_dim,window_size,n_head,n_layer,dropout,use_dyt=False,dyt_alpha=0.2)
        self.Linear_critic = nn.Linear(embd_dim,1)

    def step(self,s):
        logits = self(s[:,-self.window_size:])

        logits=logits[:,-1,:]
        dist = F.softmax(logits,dim=-1)
        dist = Categorical(dist)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        s_next = torch.cat((s,action.unsqueeze(0)),dim=1)
        done = False
        if action == 50256:
            done=True
        return action, log_prob, s_next[:,-self.window_size:], done

    def actor_critic(self,s,attention_mask=None):
        x=s[:,-self.window_size:]
        B,T= x.shape
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(torch.arange(T,device=tok_emb.device))
        x=tok_emb+pos_emb

        for block in self.blocks:
            x=block(x,attention_mask=attention_mask)

        if self.use_dyt==True:
            x=self.dyt(x)
        else:
            x=self.ln_f(x)

        if attention_mask is not None:
            batch_indices = torch.arange(B)
            masked = attention_mask * torch.arange(T, device=attention_mask.device).unsqueeze(-1)
            token_indices = masked.argmax(dim=1)
            x=x[batch_indices,token_indices.squeeze(-1)]
        else:
            x=x[:,-1,:]

        value = self.Linear_critic(x) # use the last token
        logits = self.lm_head(x)
        dist = F.softmax(logits,dim=-1)
        dist = Categorical(dist)

        return dist,value
