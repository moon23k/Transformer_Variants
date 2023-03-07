import os, torch
import torch.nn as nn
from model.original import OriginalTransformer
from model.recurrent import RecurrentTransformer
from model.evolved import EvolvedTransformer



def init_xavier(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


def count_params(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params
    

def check_size(model):
    param_size, buffer_size = 0, 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def load_model(config):
    if config.model == 'original':
        model = OriginalTransformer(config)
    elif config.model == 'recurrent':
        model = RecurrentTransformer(config)
    elif config.model == 'evolved':
        model = EvolvedTransformer(config)

    if config.task != 'train':
        model_state = torch.load()
        continue

    return model.to(config.device)