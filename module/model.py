import os, torch
import torch.nn as nn
from model.original import VanillaTransformer
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
    if config.model == 'vanilla':
        model = VanillaTransformer(config)
    elif config.model == 'recurrent':
        model = RecurrentTransformer(config)
    elif config.model == 'evolved':
        model = EvolvedTransformer(config)

    model.apply(init_xavier)
    print(f"Initialized {config.model_type} model for task has loaded")

    if config.mode != 'train':
        assert os.path.exists(config.ckpt)
        model_state = torch.load(config.ckpt, map_location=config.device)['model_state_dict']
        model.load_state_dict(model_state)
        print(f"Trained Model states has loaded from {config.ckpt}")       
    
    print(f"--- Model Params: {count_params(model):,}")
    print(f"--- Model  Size : {check_size(model):.3f} MB\n")


    return model.to(config.device)