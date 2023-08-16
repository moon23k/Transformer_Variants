import copy, math, torch
import torch.nn as nn
from collections import namedtuple




def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def shift_trg(x):
    return x[:, :-1], x[:, 1:]


def generate_square_subsequent_mask(sz):
    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)


class PositionalEncoding(nn.Module):
    def __init__(self, config, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(config.dropout_ratio)
        
        pe = torch.zeros(max_len, config.emb_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.emb_dim, 2) * -(math.log(10000.0) / config.emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)



class Embeddings(nn.Module):
    def __init__(self, config):
        super(Embeddings, self).__init__()

        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.scale = math.sqrt(config.emb_dim)
        self.pos_emb = PositionalEncoding(config)
        self.dropout = nn.Dropout(config.dropout_ratio)

        self.use_fc_layer = (config.emb_dim != config.hidden_dim)
        if self.use_fc_layer:
            self.fc = nn.Linear(config.emb_dim, config.hidden_dim)


    def forward(self, x):
        out = self.tok_emb(x) * self.scale
        out = self.pos_emb(out)

        if self.use_fc_layer:
            return self.dropout(self.fc(out))
        return self.dropout(out)




class ModelBase(nn.Module):
    def __init__(self, config):
        super(ModelBase, self).__init__()
        
        self.pad_id = config.pad_id
        self.device = config.device
        self.vocab_size = config.vocab_size

        self.generator = nn.Linear(config.hidden_dim, config.vocab_size)
        self.criterion = nn.CrossEntropyLoss()
        
        self.out = namedtuple('Out', 'logit loss')
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=config.pad_id, 
            label_smoothing=0.1
        ).to(self.device)


    def encode(self):
        pass


    def decode(self):
        pass


    def forward(self, src, trg):
        trg, label = shift_trg(trg)

        src_pad_mask = (src == self.pad_id)
        trg_pad_mask = (trg == self.pad_id)
        trg_mask = generate_square_subsequent_mask(trg.size(1)).to(self.device)

        memory = self.encode(src, src_pad_mask)
        dec_out = self.decode(trg, memory, trg_mask, trg_pad_mask, src_pad_mask)
        logit = self.generator(dec_out)
        
        self.out.logit = logit
        self.out.loss = self.criterion(
            logit.contiguous().view(-1, self.vocab_size), 
            label.contiguous().view(-1)
        )

        return self.out