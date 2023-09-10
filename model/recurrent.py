import numpy as np
import torch, math
import torch.nn as nn




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
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.embedding = nn.Embedding(config.emb_dim)
        
        self.time_signal = generate_signal(
            512, config.hidden_dim
        ).to(config.device)

        self.pos_signal = generate_signal(
            config.n_layers, config.hidden_dim
        ).to(config.device)
        

        self.layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.pff_dim,
            dropout=config.dropout_ratio,
            batch_first=True
        )
        

    def forward(self, x, e_mask):
        x = self.embedding(x)
        seq_len = x.size(1)

        for l in range(self.n_layers):
            x += self.time_signal[:, :seq_len, :]
            x += self.pos_signal[:, l, :].unsqueeze(1).repeat(1, seq_len, 1)
            x = self.layer(x, src_key_padding_mask=e_mask)
        
        return self.norm(x)




class RecurrentDecoder(nn.Module):
    def __init__(self, config):
        super(RecurrentDecoder, self).__init__()    

        self.n_layers = config.n_layers
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.embedding = nn.Embedding(config.emb_dim)
        
        self.time_signal = generate_signal(
            512, config.hidden_dim
        ).to(config.device)
        
        self.pos_signal = generate_signal(
            config.n_layers, config.hidden_dim
        ).to(config.device)

        self.layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.pff_dim,
            dropout=config.dropout_ratio,
            batch_first=True
        )



    def forward(self, x, m, e_mask, d_mask):
        x = self.embedding(x)
        seq_len = x.size(1)

        for l in range(self.n_layers):
            x += self.time_signal[:, :seq_len, :]
            x += self.pos_signal[:, l, :].unsqueeze(1).repeat(1, seq_len, 1)
            x = self.layer(
                tgt=x, memory=m,
                memory_key_padding_mask=e_mask, 
                tgt_mask=d_mask
            )

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


    def pad_mask(self, x):
        return x == self.pad_mask


    def dec_mask(self, x):
        return

        
    def forward(self, x, y):
        e_mask = self.pad_mask(x)
        d_mask = self.dec_mask(y)
        
        memory = self.encoder(x, e_mask)
        dec_out = self.decoder(y, memory, e_mask, d_mask)
        
        return self.generator(dec_out)

