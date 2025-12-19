import torch
import torch.nn as nn
import math

from src.components.attention.multi_head_attention import MultiHeadAttention
from src.components.network.feed_forward_network import FeedForwardNetwork

class Encoder_Layer(nn.Module):
    def __init__(self,emb_dim,num_heads,ffn_hidden,drop_prob=0.1):
        super().__init__()
        self.attention=MultiHeadAttention(emb_dim=emb_dim,num_heads=num_heads)
        self.norm1=nn.LayerNorm(emb_dim)
        self.dropout1=nn.Dropout(p=drop_prob)
        self.ffn=FeedForwardNetwork(emb_dim,hidden_dim=ffn_hidden,drop_prob=drop_prob)
        self.norm2=nn.LayerNorm(emb_dim)
        self.dropout2=nn.Dropout(p=drop_prob)


    def forward(self,x):
        residual_x=x
        x=self.attention(x)
        x=self.dropout1(x)
        x=self.norm1(x+residual_x)
        residual_x=x
        x=self.ffn(x)
        x=self.dropout2(x)
        x=self.norm2(x+residual_x)
        return x
        





