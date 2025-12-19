import torch.nn as nn
import torch

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_sequence_length):
        super().__init__()

        even_i = torch.arange(0, emb_dim, 2).float()
        denominator = torch.pow(10000, even_i / emb_dim)

        position = torch.arange(max_sequence_length).unsqueeze(1)

        even_PE = torch.sin(position / denominator)
        odd_PE  = torch.cos(position / denominator)

        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)

        PE = PE.unsqueeze(0)  # 🔑 [1, max_sequence_length, emb_dim]

        self.register_buffer("PE", PE)

    def forward(self,x):
        return x+self.PE
