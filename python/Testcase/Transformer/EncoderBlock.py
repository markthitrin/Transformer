import torch
import torch.nn as nn
import math
import numpy as np
from config import get_config
from LayerNormalization import LayerNormalization
from MultiheadAttention import MultiHeadAttentionBlock
from FeedForwardBlock import FeedForwardBlock

class ManualDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.scale = 1.0 - dropout

    def forward(self, x):
        return x / self.scale

class ResidualConnection(nn.Module):
    
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = ManualDropout(dropout)
        self.norm = LayerNormalization(features)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=get_config()["lr"], betas=(get_config()["beta1"], get_config()["beta2"]))

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
    def getParam(self, prefix, dict, sublayer) :
        self.norm.getParam(prefix + ".layerNorm", dict)
        sublayer.getParam(prefix + ".sublayer", dict)

    def getForward(self, prefix, dict, sublayer) :
        batch_size = get_config()["batch_size"]
        seq = get_config()["seq"]
        d_model = self.linear_1.in_features 

        x = torch.randn(batch_size, seq, d_model)
        with torch.no_grad():
            y = self.forward(x, sublayer)

        dict[prefix + ".input"] = x.detach().numpy()
        dict[prefix + ".output"] = y.detach().numpy()

    def getOriginalParam(self, prefix, dict, sublayer) :
        self.norm.getOriginalParam(prefix + ".layerNorm", dict)
        sublayer.getOriginalParam(prefix + ".sublayer", dict)
    
    def getUpdatedParam(self, prefix, dict, sublayer) :
        self.norm.getUpdatedParam(prefix + ".layerNorm", dict)
        sublayer.getUpdatedParam(prefix + ".sublayer", dict)

    def getBackward(self, prefix, dict, sublayer) :
        batch_size = get_config()["batch_size"]
        seq = get_config()["seq"]
        hidden_size = self.alpha.shape[0]

        x = torch.randn(batch_size, seq, hidden_size)

        self.getOriginalParam(prefix, dict, sublayer)
        
        y = self.forward(x)
        loss = y.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.getUpdatedParam(prefix, dict, sublayer)
        
        dict[prefix + ".output"] = y.detach().numpy()
        dict[prefix + ".loss"] = loss.item()

        
class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])
        self.optimizer = torch.optim.Adam(self.parameters(), lr=get_config()["lr"], betas=(get_config()["beta1"], get_config()["beta2"]))

    def forward(self, x, src_mask, prefix=None, dict=None):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        if(prefix != None) :
            dict[prefix + ".sub1.output"] = x.detach().numpy()
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
    def getParam(self, prefix, dict) :
        self.residual_connections[0].getParam(prefix + ".sub1", dict, self.self_attention_block)
        self.residual_connections[1].getParam(prefix + ".sub2", dict, self.feed_forward_block)

    def getForward(self, prefix, dict) :
        batch_size = get_config()["batch_size"]
        seq = get_config()["seq"]
        d_model = self.self_attention_block.d_model

        x = torch.randn(batch_size, seq, d_model)
        src_mask = torch.ones(seq, seq)
        with torch.no_grad():
            y = self.forward(x, src_mask)

        dict[prefix + ".input"] = x.detach().numpy()
        dict[prefix + ".output"] = y.detach().numpy()

    def getOriginalParam(self, prefix, dict) :
        self.residual_connections[0].getOriginalParam(prefix + ".sub1", dict, self.self_attention_block)
        self.residual_connections[1].getOriginalParam(prefix + ".sub2", dict, self.feed_forward_block)
    
    def getUpdatedParam(self, prefix, dict) :
        self.residual_connections[0].getUpdatedParam(prefix + ".sub1", dict, self.self_attention_block)
        self.residual_connections[1].getUpdatedParam(prefix + ".sub2", dict, self.feed_forward_block)

    def getBackward(self, prefix, dict) :
        batch_size = get_config()["batch_size"]
        seq = get_config()["seq"]
        hidden_size = self.self_attention_block.d_model

        src_mask = torch.ones(seq, seq)
        x = torch.randn(batch_size, seq, hidden_size)

        self.getOriginalParam(prefix, dict)
        
        y = self.forward(x, src_mask)
        loss = y.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.getUpdatedParam(prefix, dict)
        
        dict[prefix + ".output"] = y.detach().numpy()
        dict[prefix + ".loss"] = loss.item()