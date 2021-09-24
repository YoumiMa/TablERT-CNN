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
        
        def row_2Datt(inputs, padding_mask):
            max_len = inputs.shape[1]
            z_row = inputs.reshape(max_len, -1, self.hid_dim)
            z_row, weights = self.attention(z_row, z_row, z_row, key_padding_mask = padding_mask)
#             print("weights:", weights)
            return z_row.view(-1, max_len, max_len, self.hid_dim)
    
        def col_2Datt(inputs, padding_mask):
            z_col = inputs.transpose(1,2)
            z_col= row_2Datt(z_col, padding_mask)
            z_col = z_col.transpose(1,2)
            return z_col

        def CR_2Datt(inputs, padding_mask):
#             print("row attention")
            z_row = row_2Datt(inputs, padding_mask)
#             print("=" * 20)
#             print("column attention")
            z_col = col_2Datt(inputs, padding_mask)
#             print("=" * 20)
            outputs = (z_col + z_row)/2.
            return outputs
        
        
        z_hat = CR_2Datt(inputs, padding_mask)
        
        return z_hat
    

class TableTransformerLayer(nn.Module):
    
    def __init__(self, n_heads, input_dim, hid_dim, attn_type, device, dropout=0.1, activation="relu"):
        
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
        

        self.layernorm = nn.LayerNorm(hid_dim)
        self.activation = _get_activation_fn(activation)
        self.attn_type = attn_type
        
    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TableTransformerLayer, self).__setstate__(state)
    

    def forward(self, inputs, src_mask: Optional[torch.tensor] = None, 
                src_key_padding_mask: Optional[torch.tensor] = None) -> torch.tensor:
        
        if self.attn_type == "CR":
            z_hat = self.attn(inputs, padding_mask = src_key_padding_mask)
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
#         reporter = MemReporter(self.attn)
#         reporter.report()
        return outputs


        
class TableTransformer(nn.Module):
    
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers=3, norm=None):
        
        super(TableTransformer, self).__init__()
        
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: torch.tensor, mask: Optional[torch.tensor] = None, 
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
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output
