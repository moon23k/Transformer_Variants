import random, argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import numpy as np
import sentencepiece as spm

from module.model import load_model
from module.data import load_dataloader

from module.train import Trainer
from module.test import Tester
from module.search import Search



class Config(object):
    def __init__(self, args):    

        self.task = args.task
        self.mode = args.mode
        self.model_type = args.model
        self.ckpt = f"ckpt/{self.task}/{self.model_type}.pt"

        #Training Configs
        self.n_epochs = 10
        self.batch_size = 128
        self.lr = 1e-4
        self.clip = 1
        self.early_stop = True
        self.patience = 3
        self.iters_to_accumulate = 4

        #Model Configs
        self.emb_dim = 256
        self.hidden_dim = 512
        self.pff_dim = 2048
        self.n_heads = 8
        self.n_layers = 3
        self.dropout_ratio = 0.1

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.mode == 'inference':
            self.search_method = args.search
            self.device = torch.device('cpu')

    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")



def set_seed(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True


def load_tokenizer(task):
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(f'data/{task}/tokenizer.model')
    tokenizer.SetEncodeExtraOptions('bos:eos')    
    return tokenizer    



def inference(config, model, tokenizer):
    search_module = Search(config, model)

    print(f'--- Inference Process Started! ---')
    print('[ Type "quit" on user input to stop the Process ]')
    
    while True:
        input_seq = input('\nUser Input Sequence >> ').lower()

        #Enc Condition
        if input_seq == 'quit':
            print('\n--- Inference Process has terminated! ---')
            break        

        if config.search_method == 'beam':
            output_seq = search_module.beam_search(input_seq)
        else:
            output_seq = search_module.greedy_search(input_seq)

        print(f"Model Out Sequence >> {output_seq}")       




def main(args):
    set_seed()
    config = Config(args)

    tokenizer = load_tokenizer(config.task)
    setattr(config, 'pad_id', tokenizer.pad_id())
    setattr(config, 'vocab_size', tokenizer.vocab_size())
    model = load_model(config)

    if config.mode == 'train':
        train_dataloader = load_dataloader(config, 'train')
        valid_dataloader = load_dataloader(config, 'valid')
        trainer = Trainer(config, model, train_dataloader, valid_dataloader)
        trainer.train()
    elif config.mode == 'test':
        test_dataloader = load_dataloader(config, 'test')
        tester = Tester(config, model, tokenizer, test_dataloader)
        tester.test()
    elif config.mode == 'inference':
        inference(config, model, tokenizer)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', required=True)
    parser.add_argument('-mode', required=True)
    parser.add_argument('-model', required=True)
    
    args = parser.parse_args()
    assert args.mode in ['nmt', 'dialog', 'sum']
    assert args.mode in ['train', 'test', 'inference']
    assert args.model in ['vanilla', 'recurrent', 'evolved']
    
    main(args)