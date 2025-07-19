import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model) 

class PositionalEncoding(nn.Module):

    def __init__(self,d_model : int, seq_len : int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len,d_model)
        pos = torch.arange(0,d_model,dtype=torch.float).unsqueeze(1) # arange -> Tensor with range of nums -> unsqueeze adds a dim of size 1 at the specified axis making shape d_shape,1
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model)) # For numerical stability calculation is done like this in logspace
        pe[:,0::2] = torch.sin(pos*div_term)
        pe[:,1::2] = torch.cos(pos*div_term)

        pe = pe.unsqueeze(0) # (1,seq_len,d_model) to support batched input

        self.register_buffer('pe',pe) #If we want to save the tensor along with model save it to buffer

    def forward(self,x):
        x = x + (self.pe[:,:x.shape(1),:]).requires_grad_(False) #Pe is broadcasted across the batch
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps:float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        mean = x.mean(dim=-1,keepdim=True) #mean is done across last dim -> d_model, keepdim = True keeps the last dim to 1 so when we are sub at end in x-mean we can broadcast to all x
        std = x.std(dim =-1,keepdim=True)
        return self.alpha * (x-mean)/(std+self.eps)+self.bias
    

class FeedForwardNet(nn.Module):
    def __init__(self, d_model:int,d_ffs:int,dropout:float):
        super().__init__()
        self.linear1 = nn.Linear(d_model,d_ffs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffs,d_model)

    def forward(self,x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int,h : int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        
        assert d_model % h == 0,"d_model must be div by h"

        self.dropout = nn.Dropout(dropout)
        self.d_k = d_model//h
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)

        self.w_o = nn.Linear(d_model,d_model)

    
    @staticmethod
    def attention(q,k,v,mask,dropout:nn.Dropout):
        dk = q.shape[-1]
        attention_scores = (q @ k.transpose(-2,-1))/math.sqrt(dk)
        if mask:
            attention_scores.masked_fill_(mask==0,-1e9)
        attention_scores = attention_scores.softmax(dim = -1)

        if dropout:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ v) , attention_scores



    def forward(self,q,k,v,mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
       
        #.view() to reshape

        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2) #Batch,seq_len,d_model -> Batch,seq_len,h,d_k -> Batch,h,seq_len,dk
        key = key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value =value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)

        x,self.attention_scores = MultiHeadAttention.attention(query,key,value,mask,self.dropout)
        
       
       