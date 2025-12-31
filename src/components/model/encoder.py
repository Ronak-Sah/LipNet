import torch.nn as nn
from src.components.layers.encoder_layer import Encoder_Layer
from src.components.network.positional_encoding import PositionalEncoding

class Sequential_Encoder(nn.Module):
    def __init__(self, emb_dim, num_heads, ffn_hidden, drop_prob, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([Encoder_Layer(emb_dim, num_heads,ffn_hidden, drop_prob)
                                     for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    


class Encoder(nn.Module):
    def __init__(self, ffn_hidden, num_heads, num_layers,max_len,drop_prob=0.5, emb_dim=256):
        super().__init__()
        
        self.position_encoding=PositionalEncoding(emb_dim,max_len)
        self.layers = Sequential_Encoder(emb_dim, num_heads, ffn_hidden, drop_prob, num_layers)
        self.ln = nn.LayerNorm(emb_dim)
    def forward(self, x):
        x = self.ln(x)
        x=self.position_encoding(x)
        x = self.layers(x)
        return x