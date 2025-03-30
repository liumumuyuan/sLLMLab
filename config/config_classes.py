from dataclasses import dataclass
import importlib
import yaml

class TokenizerWrapper():
    def __init__(self,tool,version,pad_token_id,eos_token_id):
        tool = importlib.import_module(tool)
        self.tokenizer = tool.get_encoding(version)
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.n_vocab=self.tokenizer.n_vocab

    def __len__(self):
        return self.n_vocab

    def encode(self,text):
        return self.tokenizer.encode(text)

    def decode(self,tokens):
        return self.tokenizer.decode(tokens)

#########################################################################
@dataclass
class DataPreparationConfig:
    pretraining_data_folder: str
    pretraining_train_data: str
    pretraining_vali_data: str

    sft_data_folder: str
    sft_train_data: str
    sft_vali_data: str

    reward_model_data_folder: str
    reward_model_train_data: str
    reward_model_vali_data: str

    RLHF_data_folder: str
    RLHF_train_data: str
    RLHF_vali_data: str

@dataclass
class TokenizerConfig:
    tool: str
    version: str
    pad_token_id: int
    eos_token_id: int

    tokenizer: TokenizerWrapper = None
    #n_vocab: int = None
    def __post_init__(self): #post initialization
        self.tokenizer = TokenizerWrapper(self.tool,self.version,self.pad_token_id,self.eos_token_id)
        #self.n_vocab = self.tokenizer.n_vocab
    @property
    def n_vocab(self):
        return self.tokenizer.n_vocab

@dataclass
class ModelConfig:
    seed: int
    device: str
    n_layer: int
    n_head: int
    embd_dim: int
    window_size: int
    dropout: float
    use_dyt: bool
    dyt_alpha: float
    max_new_tokens: int

@dataclass
class PretrainingConfig:
    lr: float
    batch_size: int
    epochs: int
    eval_iters: int
    eval_itvl: int
    folder: str
    model_path: str
    loss_path: str
    data_folder: str
    train_data: str
    vali_data: str

@dataclass
class SFTConfig:
    lr: float
    batch_size: int
    epochs: int
    eval_iters: int
    eval_itvl: int
    data_folder: str
    train_data: str
    vali_data: str
    folder: str
    model_init_path: str
    model_path: str
    loss_path: str

@dataclass
class RWMConfig:
    lr: float
    batch_size: int
    epochs: int
    eval_iters: int
    eval_itvl: int
    max_prompted_tokens: int
    K: int
    host_ip: str
    url: str
    payload: dict
    prompt: str
    folder: str
    data_folder: str
    train_data: str
    vali_data: str
    prompting_model_path: str
    model_init_path: str
    model_path: str
    loss_path: str

@dataclass
class RLHFConfig:
    lr: float
    inner_batch_size: int
    epochs: int
    eval_itvl: int
    minsize_experience_buffer: int
    N: int
    gamma: float
    gae_lambda: float
    policy_clip: float
    epochs_experience: int
    max_len_episode: int
    folder: str
    train_data: str
    vali_data: str
    sft_model_path: str
    reward_model_path: str
    actor_critic_model_init_path: str
    result_model_path: str
    scores_path: str

@dataclass
class ChatConfig:
    model_path: str
    max_new_tokens: int
###########################################################################

@dataclass
class FullConfig:
    dprp: DataPreparationConfig
    tknz: TokenizerConfig
    hypp: ModelConfig
    prt: PretrainingConfig
    sft: SFTConfig
    rwm: RWMConfig
    rlhf: RLHFConfig
    inference: ChatConfig

def load_config(path: str) ->FullConfig:
    with open(path,'r') as f:
        raw  = yaml.safe_load(f)
    return FullConfig(
        dprp = DataPreparationConfig(**raw['dprp']),
        tknz = TokenizerConfig(**raw['tknz']),
        hypp = ModelConfig(**raw['hypp']),
        prt  = PretrainingConfig(**raw['prt']),
        sft  = SFTConfig(**raw['sft']),
        rwm  = RWMConfig(**raw['rwm']),
        rlhf = RLHFConfig(**raw['rlhf']),
        inference = ChatConfig(**raw['inference'])
    )
