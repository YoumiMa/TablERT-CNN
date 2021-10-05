import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import numpy as np

from torch import Tensor
from typing import Optional, Any, Tuple
from pytorch_memlab import profile, MemReporter

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    

class CR_2DMultiheadAttention(nn.Module):
    
    ''' Cross-Road 2D multihead attention.'''
    def __init__(self, n_heads, hid_dim, dropout, device):
        
        super(CR_2DMultiheadAttention, self).__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.device = device
        
        self.attention = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=n_heads, dropout=dropout)
        

    def forward(self, inputs, attn_mask = None, padding_mask = None):
        
        def col_2Datt(inputs, padding_mask):
            
            pad_mask =  padding_mask.unsqueeze(1) * padding_mask.unsqueeze(1).permute(0,2,1)
            pad_mask += torch.eye(pad_mask.shape[-1], dtype=torch.bool, device = self.device)
            pad_mask = (~pad_mask).flatten(start_dim=0, end_dim=1)   
            
            max_len = inputs.shape[1]
            z_col = inputs.reshape(max_len, -1, self.hid_dim)
            z_col, weights = self.attention(z_col, z_col, z_col, key_padding_mask = pad_mask)
            return z_col.view(-1, max_len, max_len, self.hid_dim), weights
    
        def row_2Datt(inputs, padding_mask):
            
            z_row = inputs.transpose(1,2)
            z_row, weights = col_2Datt(z_row, padding_mask)
            z_row = z_row.transpose(1,2)
            
            return z_row, weights

        def diag_att(inputs, padding_mask):
            
            batch_size, max_len = inputs.shape[0], inputs.shape[1]
            z_diag = inputs.flatten(start_dim=1,end_dim=2).transpose(0,1)
            diag_repr = inputs.diagonal(dim1=1, dim2=2).permute(2,0,1)
 
            z_diag, weights = self.attention(z_diag, diag_repr, diag_repr)
            return z_diag.view(-1, max_len, max_len, self.hid_dim), weights
        
        def CR_2Datt(inputs, padding_mask):
            
            
            z_col, weights_col = col_2Datt(inputs, padding_mask)
#             print("=" * 20)
            z_row, weights_row = row_2Datt(inputs, padding_mask)
#             print("=" * 20)
#             z_diag, weights_diag = diag_att(inputs, padding_mask)
    
            torch.save({'row': weights_row,
                       'col': weights_col}, 'weights_padded')
            outputs = z_col
            
            return outputs      

        z_hat = CR_2Datt(inputs, padding_mask)
        
        return z_hat
    

class TableTransformerLayer(nn.Module):
    
    def __init__(self, n_heads, input_dim, hid_dim, attn_type, device, dropout=0.3, activation="relu"):
        
        super(TableTransformerLayer, self).__init__()
        
        if attn_type == "CR":
            self.attn = CR_2DMultiheadAttention(n_heads, input_dim, dropout, device)
        else:
            self.attn = nn.MultiHeadAttention(embed_dim=input_dim, num_heads=n_heads, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(input_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, input_dim)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        

        self.activation = _get_activation_fn(activation)
        self.attn_type = attn_type
        
    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TableTransformerLayer, self).__setstate__(state)
    

    def forward(self, inputs, src_mask: Optional[torch.tensor] = None, 
                src_key_padding_mask: Optional[torch.tensor] = None) -> torch.tensor:
        
        if self.attn_type == "CR":
            z_hat = self.attn(inputs, padding_mask = src_key_padding_mask, attn_mask = src_mask)
        else:
#             print("inputs:", inputs.shape)
            z_hat, weights = self.attn(inputs, inputs, inputs)

#             torch.save(weights, 'tttt')
#             print("weights:", weights[0])
            
        inputs = inputs + self.dropout1(z_hat)
        inputs = self.norm1(inputs)
        outputs_hat = self.linear2(self.dropout(self.activation(self.linear1(inputs))))
        outputs = inputs + self.dropout2(outputs_hat)
        outputs = self.norm2(outputs)
# #         reporter = MemReporter(self.attn)
#         reporter.report()
        return outputs


        
class TableTransformer(nn.Module):
    
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers=3, norm=None):
        
        super(TableTransformer, self).__init__()
        
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: torch.tensor, src_mask: Optional[torch.tensor] = None, 
                src_key_padding_mask: Optional[torch.tensor] = None) -> torch.tensor:
        """Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output
