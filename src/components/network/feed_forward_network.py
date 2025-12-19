import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    def __init__(self,emb_dim,hidden_dim,drop_prob=0.1):
        super().__init__()
        self.linear1=nn.Linear(emb_dim,hidden_dim)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(p=drop_prob)
        self.linear2=nn.Linear(hidden_dim,emb_dim)

    def forward(self,x):
        x=self.linear1(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.linear2(x)
        return x
        

