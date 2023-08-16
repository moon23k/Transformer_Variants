import math, torch
import torch.nn as nn
from collections import namedtuple
from .common import *



class StandardEncoder(nn.Module):
    def __init__(self, config):
        super(StandardEncoder, self).__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.pff_dim,
            dropout=config.dropout_ratio,
            activation='gelu',
            batch_first=True
        )

        self.embeddings = Embeddings(config)
        self.layers = clones(layer, config.n_layers)


    def forward(self, x, src_key_padding_mask):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
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
            batch_first=True
        )

        self.embeddings = Embeddings(config)
        self.layers = clones(layer, config.n_layers)


    def forward(self, x, memory, tgt_mask, 
                tgt_key_padding_mask, 
                memory_key_padding_mask):

        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(
                x, memory, 
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
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
        
        self.out = namedtuple('Out', 'logit loss')
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=config.pad_id, 
            label_smoothing=0.1
        ).to(self.device)
        
        
    def forward(self, src, trg):
        trg, label = shift_trg(trg)

        #Masking
        src_pad_mask = src == self.pad_id
        trg_pad_mask = trg == self.pad_id
        trg_mask = generate_square_subsequent_mask(trg.size(1)).to(self.device)
        
        #Actual Processing
        memory = self.encoder(src, src_key_padding_mask=src_pad_mask)
        dec_out = self.decoder(
            trg, memory, tgt_mask=trg_mask, 
            tgt_key_padding_mask=trg_pad_mask, 
            memory_key_padding_mask=src_pad_mask
        )
        logit = self.generator(dec_out)
        

        #Getting Outputs
        self.out.logit = logit
        self.out.loss = self.criterion(
            logit.contiguous().view(-1, self.vocab_size), 
            label.contiguous().view(-1)
        )
        
        return self.out
