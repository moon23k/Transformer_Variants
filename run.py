import random, torch, yaml, argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import sentencepiece as spm

import models.base as base
import models.recurrent as recurrent
import models.evolved as evolved

from modules.data import load_dataloader
from modules.train import Trainer
from modules.test import Tester
from modules.inference import Translator



def set_seed(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True


def load_tokenizer(lang):
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(f'data/{lang}_spm.model')
    tokenizer.SetEncodeExtraOptions('bos:eos')    
    return tokenizer    


def init_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)    


def count_params(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model params: {params:,}")


def check_size(model):
    param_size, buffer_size = 0, 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))


def load_model(config):
    if config.model_name == 'base':
        model = base.Transformer(config).to(config.device)
    elif config.model_name == 'recurrent':
        model = recurrent.Transformer(config).to(config.device)
    elif config.model_name == 'evolved':
        model = evolved.Transformer(config).to(config.device)     


    if config.task == 'train':
        model.apply(init_weights)
    else:
        model_state = torch.load(config.ckpt_path, map_location=config.device)['model_state_dict']
        model.load_state_dict(model_state)
        model.eval()

    return model


class Config:
    def __init__(self, task, model_name):
        with open('configs/model.yaml', 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)        
            for p in params.items():
                setattr(self, p[0], p[1])
        
        self.task = task
        self.model_name = model_name

        self.unk_idx = 0
        self.pad_idx = 1
        self.bos_idx = 2
        self.eos_idx = 3
        
        self.clip = 1
        self.n_epochs = 10
        self.batch_size = 128
        self.learning_rate = 5e-4
        self.scheduler = 'constant'
        self.ckpt_path = f'ckpt/{model_name}.pt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def print_attr(self):
        for attr, val in self.__dict__.items():
            print(f"* {attr}: {val}")


def main(config):
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', required=True)
    parser.add_argument('-model', required=True)
    parser.add_argument('-scheduler', default='constant', required=False)
    
    args = parser.parse_args()
    assert args.task in ['train', 'test', 'inference']
    assert args.model in ['seq2seq', 'attention', 'transformer']
    
    set_seed()
    config = Config(args)
    main(config)