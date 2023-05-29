import numpy as np
import torch, math
import torch.nn as nn
from collections import namedtuple
from model.common import (Embeddings, shift_trg, 
                          generate_square_subsequent_mask)



def generate_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = ( math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * np.exp(np.arange(num_timescales).astype(np.float) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]], 'constant', constant_values=[0.0, 0.0])
    signal =  signal.reshape([1, length, channels])

    return torch.from_numpy(signal).type(torch.FloatTensor)


class RecurrentEncoder(nn.Module):
    def __init__(self, config):
        super(RecurrentEncoder, self).__init__()    

        self.n_layers = config.n_layers
        self.embeddings = Embeddings(config)
        self.time_signal = generate_signal(512, config.hidden_dim).to(config.device)
        self.pos_signal = generate_signal(config.n_layers, config.hidden_dim).to(config.device)
        self.layer = nn.TransformerEncoderLayer(d_model=config.hidden_dim,
                                                nhead=config.n_heads,
                                                dim_feedforward=config.pff_dim,
                                                dropout=config.dropout_ratio,
                                                batch_first=True)
        self.norm = nn.LayerNorm(config.hidden_dim)


    def forward(self, x, x_pad_mask):
        x = self.embeddings(x)
        seq_len = x.size(1)

        for l in range(self.n_layers):
            x += self.time_signal[:, :seq_len, :]
            x += self.pos_signal[:, l, :].unsqueeze(1).repeat(1, seq_len, 1)
            x = self.layer(x, src_key_padding_mask=x_pad_mask)
        
        return self.norm(x)



class RecurrentDecoder(nn.Module):
    def __init__(self, config):
        super(RecurrentDecoder, self).__init__()    

        self.n_layers = config.n_layers
        self.embeddings = Embeddings(config)
        self.time_signal = generate_signal(512, config.hidden_dim).to(config.device)
        self.pos_signal = generate_signal(config.n_layers, config.hidden_dim).to(config.device)
        self.layer = nn.TransformerDecoderLayer(d_model=config.hidden_dim,
                                                nhead=config.n_heads,
                                                dim_feedforward=config.pff_dim,
                                                dropout=config.dropout_ratio,
                                                batch_first=True)
        self.norm = nn.LayerNorm(config.hidden_dim)


    def forward(self, x, m, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask):
        x = self.embeddings(x)
        seq_len = x.size(1)

        for l in range(self.n_layers):
            x += self.time_signal[:, :seq_len, :]
            x += self.pos_signal[:, l, :].unsqueeze(1).repeat(1, seq_len, 1)
            x = self.layer(tgt=x, memory=m, 
                           tgt_mask=tgt_mask, 
                           tgt_key_padding_mask=tgt_key_padding_mask, 
                           memory_key_padding_mask=memory_key_padding_mask)
        return self.norm(x)



class RecurrentTransformer(nn.Module):
    def __init__(self, config):
        super(RecurrentTransformer, self).__init__()
        
        self.pad_id = config.pad_id
        self.device = config.device
        self.vocab_size = config.vocab_size
        
        self.encoder = RecurrentEncoder(config)
        self.decoder = RecurrentDecoder(config)

        self.generator = nn.Linear(config.hidden_dim, config.vocab_size)
        self.criterion = nn.CrossEntropyLoss()
        self.out = namedtuple('Out', 'logit loss')

        
    def forward(self, src, trg):
        trg, label = shift_trg(trg)
        
        src_pad_mask = (src == self.pad_id)
        trg_pad_mask = (trg == self.pad_id)
        trg_mask = generate_square_subsequent_mask(trg.size(1)).to(self.device)

        memory = self.encoder(src, src_key_padding_mask=src_pad_mask)
        dec_out = self.decoder(trg_emb, memory, tgt_mask=trg_mask, 
                               tgt_key_padding_mask=trg_pad_mask, 
                               memory_key_padding_mask=src_pad_mask)
        logit = self.generator(dec_out)
        
        self.out.logit = logit
        self.out.loss = self.criterion(logit.contiguous().view(-1, self.vocab_size), 
                                       label.contiguous().view(-1))
        
        return self.out

