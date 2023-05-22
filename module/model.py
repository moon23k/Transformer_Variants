import os, torch
import torch.nn as nn
from model.vanilla import VanillaTransformer
from model.recurrent import RecurrentTransformer
from model.evolved import EvolvedTransformer



def init_xavier(model):
    for name, param in model.named_parameters():
        if name not in ['norm', 'bias']:
            nn.init.xavier_uniform_(p)



def print_model_desc(model):
    #Number of trainerable parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"--- Model Params: {n_params:,}")

    #Model size check
    param_size, buffer_size = 0, 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f"--- Model  Size : {size_all_mb:.3f} MB")

    #GPU Memory Occupations
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"--- GPU memory occupied: {info.used//1024**2} MB\n")




def load_model(config):
    if config.model_type == 'vanilla':
        model = VanillaTransformer(config)
    elif config.model_type == 'recurrent':
        model = RecurrentTransformer(config)
    elif config.model_type == 'evolved':
        model = EvolvedTransformer(config)

    model.apply(init_xavier)
    print(f"Initialized {config.model_type} model for task has loaded")

    if config.mode != 'train':
        assert os.path.exists(config.ckpt)
        model_state = torch.load(config.ckpt, map_location=config.device)['model_state_dict']
        model.load_state_dict(model_state)
        print(f"Trained Model states has loaded from {config.ckpt}")       
    
    print_model_desc(model)
    return model.to(config.device)