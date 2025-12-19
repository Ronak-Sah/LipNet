import torch 
import torch.nn as nn

from src.components.model.cnn_model import CnnEncoder
from src.components.model.encoder import Encoder

class Transformer(nn.Module):
    def __init__(self, ffn_hidden, num_heads, num_layers,max_length,vocab_size):
        super().__init__()
        self.cnn=CnnEncoder()
        self.encoder=Encoder(ffn_hidden=ffn_hidden,num_heads=num_heads,max_len=max_length,num_layers=num_layers)
        self.linear=nn.Linear(ffn_hidden,vocab_size)


    def forward(self,x):
        x=self.cnn(x)
        print("After CNN:", x.shape)
        x=self.encoder(x)
        x=self.cnn(x)
        print("After Encoder:", x.shape)
        out=self.linear(x)
        return out