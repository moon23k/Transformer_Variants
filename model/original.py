import torch, math
import torch.nn as nn
from model.common import Embeddings




class OriginalTransformer(nn.Module):
    def __init__(self, config):
        super(OriginalTransformer, self).__init__()
        
        self.pad_id = config.pad_id
        self.device = config.device

        self.enc_emb = Embeddings(config.vocab_size, config.hidden_dim)
        self.dec_emb = Embeddings(config.vocab_size, config.hidden_dim)
        
        self.transformer = nn.Transformer(d_model=config.hidden_dim,
                                          nhead=config.n_heads,
                                          num_encoder_layers=config.n_layers,
                                          num_decoder_layers=config.n_layers,
                                          dropout=config.dropout_ratio,
                                          batch_first=True)

        self.generator = nn.Linear(config.hidden_dim, config.vocab_size)
        self.criterion = nn.CrossEntropyLoss()
        self.out = namedtuple('Out', 'logit loss')

        
    def forward(self, src, trg, label):
        src_pad_mask = (src == self.pad_id)
        trg_pad_mask = (trg == self.pad_id)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg.size(1))

        src_emb = self.enc_emb(src)
        trg_emb = self.dec_emb(trg)

        memory = self.encode(src_emb, src_pad_mask)
        dec_out = self.decode(trg_emb, memory, trg_mask, src_pad_mask, trg_pad_mask)
        
        self.out.logit = self.generator(dec_out)
        self.out.loss = self.criterion(logit, label)
        
        return self.out


    def encode(self, src_emb, src_pad_mask):
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_pad_mask)


    def decode(self, trg_emb, memory, trg_mask, trg_pad_mask, src_pad_mask):
        return self.transformer.decoder(trg_emb, memory, tgt_mask=trg_mask,
                                        tgt_key_padding_mask=trg_pad_mask,
                                        memory_key_padding_mask=src_pad_mask)