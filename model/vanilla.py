import math, torch
import torch.nn as nn
from collections import namedtuple
from model.common import *



class VanillaEncoder(nn.Module):
    def __init__(self, config):
        super(VanillaEncoder, self).__init__()

        self.embeddings = Embeddings(config)
        layer = nn.TransformerEncoderLayer(d_model=config.hidden_dim,
                                           nhead=config.n_heads,
                                           dim_feedforward=config.pff_dim,
                                           dropout=config.dropout_ratio,
                                           activation='gelu',
                                           batch_first=True)
        self.layers = clones(layer, config.n_layers)
        self.norm = nn.LayerNorm(config.hidden_dim)


    def forward(self, x, x_pad_mask):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=x_pad_mask)
        return self.norm(x)



class VanillaDecoder(nn.Module):
    def __init__(self, config):
        super(VanillaDecoder, self).__init__()

        self.embeddings = Embeddings(config)
        layer = nn.TransformerDecoderLayer(d_model=config.hidden_dim,
                                           nhead=config.n_heads,
                                           dim_feedforward=config.pff_dim,
                                           dropout=config.dropout_ratio,
                                           activation='gelu',
                                           batch_first=True)
        self.layers = clones(layer, config.n_layers)
        self.norm = nn.LayerNorm(config.hidden_dim)


    def forward(self, x, memory, x_mask, x_pad_mask, m_pad_mask):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=x_mask,
                      tgt_key_padding_mask=x_pad_mask,
                      memory_key_padding_mask=m_pad_mask)
        return self.norm(out)



class VanillaTransformer(nn.Module):
    def __init__(self, config):
        super(VanillaTransformer, self).__init__()
        
        self.pad_id = config.pad_id
        self.device = config.device
        self.vocab_size = config.vocab_size
        
        self.encoder = VanillaEncoder(config)
        self.decoder = VanillaDecoder(config)

        self.generator = nn.Linear(config.hidden_dim, config.vocab_size)
        self.criterion = nn.CrossEntropyLoss()
        self.out = namedtuple('Out', 'logit loss')

        
        
    def forward(self, src, trg):
        trg, label = shift_trg(trg)

        #Masking
        src_pad_mask = src == self.pad_id
        trg_pad_mask = trg == self.pad_id
        trg_mask = generate_square_subsequent_mask(trg.size(1)).to(self.device)
        
        #Actual Processing
        memory = self.encoder(src, src_key_padding_mask=src_pad_mask)
        dec_out = self.decoder(trg, memory, tgt_mask=trg_mask, 
                               tgt_key_padding_mask=trg_pad_mask, 
                               memory_key_padding_mask=src_pad_mask)
        logit = self.generator(dec_out)
        

        #Getting Outputs
        self.out.logit = logit
        self.out.loss = self.criterion(logit.contiguous().view(-1, self.vocab_size), 
                                       label.contiguous().view(-1))
        
        return self.out
