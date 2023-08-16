import os, torch
import torch.nn as nn
from model import (
    StandardTransformer,
    RecurrentTransformer,
    EvolvedTransformer
)




def init_weights(model):
    for p in model.parameters():
        if p.dim() > 1:
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
    print(f"--- Model  Size : {size_all_mb:.3f} MB\n")




def load_model(config):
    if config.model_type == 'standard':
        model = StandardTransformer(config)
    elif config.model_type == 'recurrent':
        model = RecurrentTransformer(config)
    elif config.model_type == 'evolved':
        model = EvolvedTransformer(config)

    init_weights(model)
    print(f"Initialized {config.model_type} model has loaded")

    if config.mode != 'train':
        assert os.path.exists(config.ckpt)
        
        model_state = torch.load(
            config.ckpt, 
            map_location=config.device
        )['model_state_dict']
        
        model.load_state_dict(model_state)
        print(f"Model states has loaded from {config.ckpt}")       
    
    print_model_desc(model)
    return model.to(config.device)