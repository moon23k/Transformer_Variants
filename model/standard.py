import math, torch
import torch.nn as nn
from collections import namedtuple
from .common import clones, Embeddings




class StandardEncoder(nn.Module):
    def __init__(self, config):
        super(StandardEncoder, self).__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.pff_dim,
            dropout=config.dropout_ratio,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.embeddings = Embeddings(config)
        self.layers = clones(layer, config.n_layers)


    def forward(self, x, e_mask):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=e_mask)
        return x



class StandardDecoder(nn.Module):
    def __init__(self, config):
        super(StandardDecoder, self).__init__()

        layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.pff_dim,
            dropout=config.dropout_ratio,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.embeddings = Embeddings(config)
        self.layers = clones(layer, config.n_layers)


    def forward(self, x, memory, e_mask, d_mask):

        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(
                x, memory, 
                memory_key_padding_mask=e_mask,
                tgt_mask=d_mask,
            )

        return x



class StandardTransformer(nn.Module):
    def __init__(self, config):
        super(StandardTransformer, self).__init__()
        
        self.pad_id = config.pad_id
        self.device = config.device
        self.vocab_size = config.vocab_size
        
        self.encoder = StandardEncoder(config)
        self.decoder = StandardDecoder(config)
        self.generator = nn.Linear(config.hidden_dim, config.vocab_size)

        self.criterion = nn.CrossEntropyLoss()
        self.out = namedtuple('Out', 'logit loss')        


    @staticmethod    
    def shift_y(x):
        return x[:, :-1], x[:, 1:]    


    def pad_mask(self, x):
        return x == self.pad_id


    def dec_mask(self, x):
        sz = x.size(1)
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(self.device)


    def forward(self, x, y):
        y, label = self.shift_y(y)

        #Masking
        e_mask = self.pad_mask(x)
        d_mask = self.dec_mask(y)
        
        #Actual Processing
        memory = self.encoder(x, e_mask)
        dec_out = self.decoder(y, memory, e_mask, d_mask)
        logit = self.generator(dec_out)
        
        #Getting Outputs
        self.out.logit = logit
        self.out.loss = self.criterion(
            logit.contiguous().view(-1, self.vocab_size), 
            label.contiguous().view(-1)
        )
        
        return self.out
