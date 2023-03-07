import random, torch, yaml, argparse
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import sentencepiece as spm

from module.model import load_model
from module.data import load_dataloader
from module.train import Trainer
from module.test import Tester



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



class Config(object):
    def __init__(self, args):    

        with open('config.yaml', 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            for group in params.keys():
                for key, val in params[group].items():
                    setattr(self, key, val)

        self.mode = args.mode
        self.model_type = args.model
        self.max_pred_len = self.max_pred_len
        self.ckpt = f"ckpt/{self.model_type}.pt"

        use_cuda = torch.cuda.is_available()
        self.device_type = 'cuda' if use_cuda else 'cpu'
        self.device = torch.device(self.device_type)

        if self.mode == 'inference':
            self.search_method = args.search
            self.device = torch.device('cpu')

    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")



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