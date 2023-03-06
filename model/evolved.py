import copy, torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F

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
        self.dropout = nn.Dropout(config.dropout_ratio)
        self.attention = nn.MultiHeadAttention(config.hidden_dim, config.num_heads)

        self.mid_layer_norm = nn.LayerNorm(config.pff_dim)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(config.hidden_dim) for _ in range(4)])        

        self.left_branch = nn.Sequential(nn.Linear(config.hidden_dim, config.pff_dim),
                                         nn.ReLU())
        self.right_branch = nn.Sequential(nn.Conv1d(in_channels=config.hidden_dim, 
                                                    out_channels=config.hidden_dim//2, 
                                                    kernel_size=3, padding=1),
                                          nn.ReLU())

        self.sep_conv = SeparableConv1D(config.pff_dim, config.hidden_dim // 2, 9)

        self.pff = nn.Sequential(nn.Linear(config.hidden_dim, config.pff_dim),
                                 nn.ReLU(),
                                 nn.Linear(config.pff_dim, config.hidden_dim))


    def forward(self, x):
        ### Block_01
        B01_out = self.glu(self.layer_norms[0](x))


        ### Block_02
        B02_normed = self.layer_norms[1](B01_out)        

        left_out = self.left_branch(B02_normed)
        left_out = self.dropout(left_out)

        right_out = self.right_branch(B02_normed.transpose(1, 2)).transpose(1, 2)
        right_out = F.pad(input=right_branch, 
                          pad=(0, left_out.shape[2] - right_out.shape[2], 0,0,0,0), 
                          mode='constant', value=self.pad_id)
        right_branch = self.dropout(right_branch)
        
        B02_out = left_branch_out + right_branch_out


        ### Block_03
        B03_out = self.mid_layer_norm(B02_out)
        B03_out = self.sep_conv(B03_out.transpose(1, 2)).transpose(1, 2)

        B03_out = F.pad(input=B03_out,
                        pad=(0, B01_out.shape[2] - B03_out.shape[2], 0, 0, 0, 0),
                        mode='constant', value=self.pad_id)
        B03_out += B01_out


        ### Block_04
        B04_out = self.layer_norms[2](B03_out).transpose(0, 1)
        attention_out = self.attention(B04_out, B04_out, B04_out, need_weight=False)[0].transpose(0, 1)
        B04_out += attention_out


        ### Block_05 & 06
        out = self.layer_norms[3](B04_out)
        out = self.pff(out) + B04_out

        return out 


class DecoderCell(nn.Module):
    def __init__(self, config):
        super(DecoderCell, self).__init__()


    def forward(self, x):
        return out



class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.emb = Embeddings()
        #config에서 if model_type == 'evolved' config.num_cells = num_layers // 2
        self.cells = clone(EncoderCell, config.num_cells)


    def forward(self, x):
        for cell in self.cells:
            x = cell(x)
        return out



class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.emb = Embeddings()
        #config에서 if model_type == 'evolved' config.num_cells = num_layers // 2
        self.cells = clone(DecoderCell, config.num_cells)


    def forward(self, x):
        for cell in self.cells:
            x = cell(x)
        return out


class EvolvedTransformer(nn.Module):
    def __init__(self, config):
        super(EvolvedTransformer, self).__init__()

        self.encoder = Encoder(config) 
        self.decoder = Decoder(config)


    def forward(self, x):
        return out



class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        
        self.device = config.device
        self.pad_id = config.pad_id

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.fc_out = nn.Linear(config.hidden_dim, config.vocab_size)

    def pad_mask(self, x):
        return (x != self.pad_id).unsqueeze(1).unsqueeze(2)

    def dec_mask(self, x):
        seq_len = x.size(-1)
        attn_shape = (1, seq_len, seq_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8) == 0
        return self.pad_mask(x) & subsequent_mask.to(self.device)

    def forward(self, src, trg):
        e_mask, d_mask = self.pad_mask(src), self.dec_mask(trg)
        memory = self.encoder(src, e_mask)
        dec_out = self.decoder(trg, memory, e_mask, d_mask)
        return self.fc_out(dec_out)        