import numpy as np
import torch, math
import torch.nn as nn
from collections import namedtuple
from model.common import PositionalEncoding



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

        self.num_layers = config.num_layers

        self.time_signal = generate_signal()
        self.pos_signal = generate_signal()

        self.layer = nn.TransformerEncoderLayer(batch_first=True)


    def forward(self, x):
        dtype = x.dtype
        seq_len = x.size(1)


        for l in range(self.num_layers):
            x += self.time_signal
            x += self.pos_signal
            x = self.layer(x)
        
        return x



class RecurrentDecoder(nn.Module):
    def __init__(self, config):
        super(RecurrentDecoder, self).__init__()    

        self.num_layers = config.num_layers
        
        self.decoder_layer = nn.TransformerDecoderLayer(batch_first=True)

    def forward(self, x):
        return



class RecurrentTransformer(nn.Module):
    def __init__(self, config):
        super(RecurrentTransformer, self).__init__()
        
        self.pad_id = config.pad_id
        self.device = config.device
        self.vocab_size = config.vocab_size

        self.enc_emb = nn.Sequential([nn.Embeddings(config.vocab_size, config.emb_dim),
                                      nn.Linear(config.emb_dim, config.hidden_dim)])

        self.dec_emb = nn.Sequential([nn.Embeddings(config.vocab_size, config.emb_dim),
                                      nn.Linear(config.emb_dim, config.hidden_dim)])        
        
        self.encoder = RecurrentEncoder(config)
        self.decoder = RecurrentDecoder(config)

        self.generator = nn.Linear(config.hidden_dim, config.vocab_size)
        self.criterion = nn.CrossEntropyLoss()
        self.out = namedtuple('Out', 'logit loss')

        
    def forward(self, src, trg, label):
        src_pad_mask = (src == self.pad_id)
        trg_pad_mask = (trg == self.pad_id)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg.size(1))

        src_emb = self.enc_emb(src)
        trg_emb = self.enc_emb(trg)

        memory = self.encode(src_emb, src_pad_mask)
        dec_out = self.decode(trg_emb, memory, trg_mask, src_pad_mask, trg_pad_mask)
        
        self.out.logit = self.generator(dec_out)
        self.out.loss = self.criterion(logit.contiguous().view(-1, self.vocab_size), 
                                       label.contiguous().view(-1))
        
        return self.out


    def encode(self, src_emb, src_pad_mask):
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_pad_mask)


    def decode(self, trg_emb, memory, trg_mask, trg_pad_mask, src_pad_mask):
        return self.transformer.decoder(trg_emb, memory, tgt_mask=trg_mask,
                                        tgt_key_padding_mask=trg_pad_mask,
                                        memory_key_padding_mask=src_pad_mask)
            