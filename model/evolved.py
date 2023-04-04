import copy, torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F
from collections import namedtuple
from model.common import *
#This is swift act func for decoder
#torch.nn.SiLU


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


#GLU
class GatedConvolution(nn.Module):
    def __init__(self, hidden_dim, kernel_size=3, padding=1):
        super(GatedConvolution,self).__init__()
        
        self.conv = nn.Conv1d(in_channels=hidden_dim, 
                              out_channels=hidden_dim * 2,
                              kernel_size=kernel_size, padding=padding, bias=True)
        init.xavier_uniform_(self.conv.weight, gain=1)

    def forward(self,x):
        convoluted = self.conv(x.transpose(1,2)).transpose(1,2)
        out, gate = convoluted.split(int(convoluted.size(-1) / 2), -1)
        out = out * torch.sigmoid(gate)
        return out



class SeparableConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SeparableConv1D, self).__init__()

        self.depth_wise = nn.Conv1d(in_channels=in_channels,
                                    out_channels=in_channels,
                                    kernel_size=kernel_size,
                                    padding="same",
                                    groups=in_channels)
        
        self.point_wise = nn.Conv1d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1)

    def forward(self, x):
        out = self.depth_wise(x)
        out = self.point_wise(out)
        return out




class EncoderCell(nn.Module):
    def __init__(self, config):
        super(EncoderCell, self).__init__()

        self.pad_id = config.pad_id
        self.glu = GatedConvolution(config.hidden_dim)
        self.attention = nn.MultiheadAttention(config.hidden_dim, config.n_heads, batch_first=True)

        self.mid_layer_norm = nn.LayerNorm(config.pff_dim)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(config.hidden_dim) for _ in range(4)])

        self.left_net = nn.Sequential(nn.Linear(config.hidden_dim, config.pff_dim),
                                      nn.ReLU(),
                                      nn.Dropout(config.dropout_ratio))

        self.right_net = nn.Sequential(nn.Conv1d(in_channels=config.hidden_dim, 
                                                 out_channels=config.hidden_dim//2, 
                                                 kernel_size=3, padding=1),
                                       nn.ReLU(),
                                       nn.Dropout(config.dropout_ratio))

        self.sep_conv = SeparableConv1D(config.pff_dim, config.hidden_dim // 2, 9)

        self.pff = nn.Sequential(nn.Linear(config.hidden_dim, config.pff_dim),
                                 nn.ReLU(),
                                 nn.Linear(config.pff_dim, config.hidden_dim))


    def forward(self, src, src_pad_mask):
        ### Block_01
        B01_out = self.glu(self.layer_norms[0](src)) #Dim:512


        ### Block_02
        B02_normed = self.layer_norms[1](B01_out)        

        left_out = self.left_net(B02_normed)
        right_out = self.right_net(B02_normed.transpose(1, 2)).transpose(1, 2)

        right_out = F.pad(input=right_out, 
                          pad=(0, left_out.size(-1) - right_out.size(-1), 0,0,0,0), 
                          mode='constant', value=self.pad_id) #Dim:2048          

        B02_out = left_out + right_out


        ### Block_03
        B03_out = self.mid_layer_norm(B02_out)
        B03_out = self.sep_conv(B03_out.transpose(1, 2)).transpose(1, 2) #Dim:256
        B03_out = F.pad(input=B03_out,
                        pad=(0, B01_out.size(-1) - B03_out.size(-1), 0, 0, 0, 0),
                        mode='constant', value=self.pad_id)
        B03_out += B01_out #Dim:512


        ### Block_04
        B04_out = self.layer_norms[2](B03_out)
        attention_out = self.attention(B04_out, B04_out, B04_out,
                                       key_padding_mask = src_pad_mask,
                                       need_weights=False)[0]
        B04_out += attention_out #Dim:512


        ### Block_05 & 06
        out = self.layer_norms[3](B04_out)
        out = self.pff(out) + B04_out #Dim:512
        return out 


class DecoderCell(nn.Module):
    def __init__(self, config):
        super(DecoderCell, self).__init__()
        
        self.pad_id = config.pad_id
        self.dropout = nn.Dropout(config.dropout_ratio)

        self.attention = nn.MultiheadAttention(config.hidden_dim, config.n_heads)

        self.mid_layer_norm = nn.LayerNorm(config.hidden_dim * 2)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(config.hidden_dim) for _ in range(4)])        


        self.left_attn = nn.MultiheadAttention(config.hidden_dim, config.n_heads * 2, batch_first=True)
        self.right_attn = nn.MultiheadAttention(config.hidden_dim, config.n_heads, batch_first=True)

        self.left_net = nn.Sequential(SeparableConv1D(config.pff_dim, config.hidden_dim // 2, 9), 
                                      nn.ReLU())
        
        self.right_net = SeparableConv1D(config.pff_dim, config.hidden_dim // 2, 9)
        
        self.sep_conv = SeparableConv1D(config.pff_dim, config.hidden_dim // 2, 9)


        self.self_attn = nn.MultiheadAttention(config.hidden_dim, config.n_heads * 2, batch_first=True)
        self.src_attn = nn.MultiheadAttention(config.hidden_dim, config.n_heads, batch_first=True)

        self.pff = nn.Sequential(nn.Linear(config.hidden_dim, config.pff_dim),
                                 nn.ReLU(),
                                 nn.Linear(config.pff_dim, config.hidden_dim))


    def forward(self, trg, memory, trg_mask, src_pad_mask, trg_pad_mask):

        ### Block_01
        B01_out = self.layer_norms[0](src)
        B01_out = self.left_attn(B01_out) + self.right_attn(B01_out)


        ### Block_02
        B02_out = self.layer_norms[1](B01_out)
        left_out = self.left_net()
        right_out = self.right_net()

        right_out = F.pad(input=right_out, 
                          pad=(0, left_out.size(-1) - right_out.size(-1), 0,0,0,0), 
                          mode='constant', value=self.pad_id) #Dim:1024          
        B02_out = left_out + right_out #Dim: 1024


        ### Block_03
        B03_out = self.mid_layer_norm(B02_out)
        B03_out = self.sep_conv(B03_out)
        B03_out += B01_out


        ### Block_04
        B04_out = self.layer_norms[2](B03_out)
        B04_out = self.self_attn()
        B04_out += B03_out


        ### Block_05
        B05_out = self.layer_norms[3](B04_out)
        B05_out = self.src_attn()
        B05_out += B04_out        


        ### Block_06 & Block_07
        out = self.layer_norms[4](B05_out)
        out = self.pff(out) + B05_out #Dim:512        

        return out



class EvolvedEncoder(nn.Module):
    def __init__(self, config):
        super(EvolvedEncoder, self).__init__()

        self.emb = Embeddings(config)
        self.cells = clones(EncoderCell(config), config.n_layers//2)


    def forward(self, src, src_pad_mask):
        x, x_mask = src, src_pad_mask
        for cell in self.cells:
            x = cell(x, x_mask)
        return x



class EvolvedDecoder(nn.Module):
    def __init__(self, config):
        super(EvolvedDecoder, self).__init__()

        self.emb = Embeddings(config)
        self.cells = clones(DecoderCell(config), config.n_layers//2)


    def forward(self, x, memory, x_mask, memory_mask):
        for cell in self.cells:
            x = cell(x)
        return out


class EvolvedTransformer(nn.Module):
    def __init__(self, config):
        super(EvolvedTransformer, self).__init__()
        
        self.pad_id = config.pad_id
        self.device = config.device
        self.vocab_size = config.vocab_size

        self.enc_emb = Embeddings(config)
        self.dec_emb = Embeddings(config)

        self.encoder = EvolvedEncoder(config) 
        self.decoder = EvolvedDecoder(config)

        self.generator = nn.Linear(config.hidden_dim, config.vocab_size)
        self.criterion = nn.CrossEntropyLoss()
        
        self.out = namedtuple('Out', 'logit loss')


    def forward(self, src, trg):
        trg, label = shift_trg(trg)

        src_pad_mask = (src == self.pad_id)
        trg_pad_mask = (trg == self.pad_id)
        trg_mask = generate_square_subsequent_mask(trg.size(1))

        src_emb = self.enc_emb(src)
        trg_emb = self.dec_emb(trg)

        memory = self.encode(src_emb, src_pad_mask)
        dec_out = self.decode(trg_emb, memory, trg_mask, src_pad_mask, trg_pad_mask)
        logit = self.generator(dec_out)
        
        self.out.logit = logit
        self.out.loss = self.criterion(logit.contiguous().view(-1, self.vocab_size), 
                                       label.contiguous().view(-1))

        return self.out


    def encode(self, src, src_pad_mask):
        return self.encoder(src, src_pad_mask)

    def decode(self, trg, memory, trg_pad_mask, trg_mask, memory_mask):
        return self.decoder(trg, memory, trg_pad_mask, trg_mask, memory_mask)