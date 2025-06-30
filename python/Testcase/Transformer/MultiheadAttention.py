import torch
import torch.nn as nn
import math
import numpy as np
from config import get_config

class ManualDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.scale = 1.0 - dropout

    def forward(self, x):
        return x / self.scale

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = ManualDropout(dropout)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=get_config()["lr"], betas=(get_config()["beta1"], get_config()["beta2"]))

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq, d_k) --> (batch, h, seq, seq)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq, seq) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq, seq) --> (batch, h, seq, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq, d_model) --> (batch, seq, d_model)
        key = self.w_k(k) # (batch, seq, d_model) --> (batch, seq, d_model)
        value = self.w_v(v) # (batch, seq, d_model) --> (batch, seq, d_model)

        # (batch, seq, d_model) --> (batch, seq, h, d_k) --> (batch, h, seq, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq, d_k) --> (batch, seq, h, d_k) --> (batch, seq, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq, d_model) --> (batch, seq, d_model)  
        return self.w_o(x)
    
    def getParam(self, prefix, dict) :
        w_q = self.w_q.weight.detach().cpu().numpy()
        w_k = self.w_k.weight.detach().cpu().numpy()
        w_v = self.w_v.weight.detach().cpu().numpy()
        w_o = self.w_o.weight.detach().cpu().numpy().T.copy()
        dict[prefix + ".w_q"] = w_q
        dict[prefix + ".w_k"] = w_k
        dict[prefix + ".w_v"] = w_v
        dict[prefix + ".w_o"] = w_o

    def getForward(self, prefix, dict, mask=None) :
        batch_size = get_config()["batch_size"]
        seq = get_config()["seq"]
        d_model = self.d_model

        q = torch.randn(batch_size, seq, d_model)
        k = torch.randn(batch_size, seq, d_model)
        v = torch.randn(batch_size, seq, d_model)
        if(mask == None) :
            mask = torch.ones(seq, seq)

        with torch.no_grad():
            y = self.forward(q, k, v, mask)
        dict[prefix + ".q"] = q.detach().numpy()
        dict[prefix + ".k"] = k.detach().numpy()
        dict[prefix + ".v"] = v.detach().numpy()
        dict[prefix + ".output"] = y.detach().numpy()

    def getOriginalParam(self, prefix, dict) :
        original_w_q = self.w_q.weight.detach().clone().cpu().numpy()
        original_w_k = self.w_k.weight.detach().clone().cpu().numpy()
        original_w_v = self.w_v.weight.detach().clone().cpu().numpy()
        original_w_o = self.w_o.weight.detach().clone().cpu().numpy().T.copy()
        dict[prefix + ".original_w_q"] = original_w_q
        dict[prefix + ".original_w_k"] = original_w_k
        dict[prefix + ".original_w_v"] = original_w_v
        dict[prefix + ".original_w_o"] = original_w_o
    
    def getUpdatedParam(self, prefix, dict) :
        updated_w_q = self.w_q.weight.detach().clone().cpu().numpy()
        updated_w_k = self.w_k.weight.detach().clone().cpu().numpy()
        updated_w_v = self.w_v.weight.detach().clone().cpu().numpy()
        updated_w_o = self.w_o.weight.detach().clone().cpu().numpy().T.copy()
        dict[prefix + ".updated_w_q"] = updated_w_q
        dict[prefix + ".updated_w_k"] = updated_w_k
        dict[prefix + ".updated_w_v"] = updated_w_v
        dict[prefix + ".updated_w_o"] = updated_w_o

    def getBackward(self, prefix, dict, mask=None) :
        batch_size = get_config()["batch_size"]
        seq = get_config()["seq"]

        q = torch.randn(batch_size, seq, self.d_model)
        k = torch.randn(batch_size, seq, self.d_model)
        v = torch.randn(batch_size, seq, self.d_model)
        if(mask == None) :
            mask = torch.ones(seq, seq)

        self.getOriginalParam(prefix, dict)

        y = self.forward(q, k, v, mask)
        loss = y.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.getUpdatedParam(prefix, dict)
        
        dict[prefix + ".output"] = y.detach().numpy()
        dict[prefix + ".loss"] = loss.item()