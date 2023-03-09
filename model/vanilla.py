import torch, math
import torch.nn as nn
from collections import namedtuple
from model.common import Embeddings, generate_square_subsequent_mask




class VanillaTransformer(nn.Module):
    def __init__(self, config):
        super(VanillaTransformer, self).__init__()
        
        self.pad_id = config.pad_id
        self.device = config.device
        self.vocab_size = config.vocab_size

        self.enc_emb = Embeddings(config)
        self.dec_emb = Embeddings(config)
        
        self.transformer = nn.Transformer(d_model=config.hidden_dim,
                                          nhead=config.n_heads,
                                          dim_feedforward=config.pff_dim,
                                          num_encoder_layers=config.n_layers,
                                          num_decoder_layers=config.n_layers,
                                          dropout=config.dropout_ratio,
                                          batch_first=True,
                                          norm_first=True)

        self.generator = nn.Linear(config.hidden_dim, config.vocab_size)
        self.criterion = nn.CrossEntropyLoss()
        self.out = namedtuple('Out', 'logit loss')

        
    def forward(self, src, trg, label):
        src_pad_mask = (src == self.pad_id).to(self.device)
        trg_pad_mask = (trg == self.pad_id).to(self.device)
        trg_mask = generate_square_subsequent_mask(trg.size(1)).to(self.device)

        src_emb = self.enc_emb(src)
        trg_emb = self.dec_emb(trg)

        memory = self.encode(src_emb, src_pad_mask)
        dec_out = self.decode(trg_emb, memory, trg_mask, trg_pad_mask, src_pad_mask)
        logit = self.generator(dec_out)
        
        self.out.logit = logit
        self.out.loss = self.criterion(logit.contiguous().view(-1, self.vocab_size), 
                                       label.contiguous().view(-1))
        
        return self.out


    def encode(self, src_emb, src_pad_mask):
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_pad_mask)


    def decode(self, trg_emb, memory, trg_mask, trg_pad_mask, src_pad_mask):
        return self.transformer.decoder(trg_emb, memory, tgt_mask=trg_mask,
                                        tgt_key_padding_mask=trg_pad_mask,
                                        memory_key_padding_mask=src_pad_mask)