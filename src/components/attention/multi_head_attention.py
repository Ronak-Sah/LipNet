import torch
import torch.nn as nn
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiHeadAttention(nn.Module):
    def __init__(self,emb_dim,num_heads):
        super().__init__()
        assert emb_dim % num_heads == 0, "Embedding dim must be divisible by num_heads"
        self.emb_dim=emb_dim                                # Stores dimension of each word 
        self.num_heads=num_heads                            # No of self attentions running parallel
        self.head_dim=emb_dim//num_heads                    # Each word dimension in 1 self attention
        
        self.qkv_layer=nn.Linear(emb_dim,3*emb_dim)         # Splitting into q,k,v (for better performance joing q,k,v[512:512:512]
        self.linear=nn.Linear(emb_dim,emb_dim)              # This layer do not change dim but improves learning after concating of q,k,v

        
    def forward(self,x):
        batch_size,each_sen_length,emb_dim=x.shape 
        # Shape of x is [batch_size,each_sen_length,emb_dim] -> {32,100,512}
        x=self.qkv_layer(x)                                       
        # After concatenation of q,k,v-->(emb_dim+emb_dim+emb_dim) shape of x is [batch_size,each_sen_length,emb_dim*3]-> {32,100,3*512}
        x=x.reshape(batch_size,each_sen_length,self.num_heads,3*self.head_dim)
        # This line divides into no of self attention(num heads)  shape of x is [batch_size,each_sen_length,num_heads,(emb_dim*3)/num_heads] -> {32,100,8,3*64}
        x=x.permute(0,2,1,3)  
        # shape of x is [batch_size,num_heads,each_sen_length,(emb_dim*3)/num_heads] -> {32,8,100,3*64}
        query,key,value=x.chunk(3,dim=-1)
        # Dividing into q,k,v -> each dim = [batch_size,num_heads,each_sen_length,(emb_dim)/num_heads] -> {32,8,100,64}
        values=scaled_dot_attention(query,key,value)
        # dim of values = [batch_size,num_heads,each_sen_length,(emb_dim)/num_heads] -> {32,8,100,64}
        values = values.permute(0, 2, 1, 3)
        # shape of x is [batch_size,each_sen_length,num_heads,(emb_dim*3)/num_heads] -> {32,100,8,3*64}
        values=values.reshape(batch_size,each_sen_length,self.emb_dim)
        # dim of values = [batch_size,num_heads,each_sen_length,(emb_dim)/num_heads] -> {32,100,512}
        out=self.linear(values) 
        # After concatenation of all num_heads -> goes into liner network -> increase learning of all num_heads
        return out
    




def scaled_dot_attention(q,k,v):
    # dim of q/k/v = [batch_size,num_heads,each_sen_length,each_head_emb_dim(head_dim)]
    dim_k=k.size()[-1]  # dim_k = head_dim
    scaled=torch.matmul(q,k.transpose(-1,-2))/math.sqrt(dim_k)
    

    attention=torch.softmax(scaled,dim=-1)
    values=torch.matmul(attention,v)
    return values

