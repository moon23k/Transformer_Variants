import random, torch, yaml, argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import sentencepiece as spm

from modules.data import load_dataloader
from modules.train import Trainer
from modules.test import Tester



def set_seed(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True


def load_tokenizer():
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(f'data/spm.model')
    tokenizer.SetEncodeExtraOptions('bos:eos')    
    return tokenizer    




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


def main(args):
    set_seed()
    config = Config(args)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', required=True)
    parser.add_argument('-model', required=True)
    
    args = parser.parse_args()
    assert args.mode in ['train', 'test', 'inference']
    assert args.model in ['original', 'recurrent', 'evolved']
    
    main(args)