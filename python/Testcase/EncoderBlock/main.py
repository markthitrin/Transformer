import torch
import torch.nn as nn
import math
import numpy as np
from config import get_config
from LayerNormalization import LayerNormalization
from MultiheadAttention import MultiHeadAttentionBlock
from FeedForwardBlock import FeedForwardBlock
import random


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

    def forward(self, x, sublayer, prefix=None, dict=None):
        x1 = self.norm(x)
        x2 = sublayer(x1)
        x3 = self.dropout(x2)
        if(prefix != None) :
            dict[prefix + ".normOutput"] = x1.detach().numpy()
            dict[prefix + ".sublayerOutput"] = x2.detach().numpy()
            dict[prefix + ".dropoutOutput"] = x3.detach().numpy()
        return x + x3
        # return x + self.dropout(sublayer(self.norm(x)))
    
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
        
        dict[prefix + ".input"] = x.detach().numpy()
        dict[prefix + ".output"] = y.detach().numpy()
        dict[prefix + ".loss"] = loss.item()

        
class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])
        self.optimizer = torch.optim.Adam(self.parameters(), lr=get_config()["lr"], betas=(get_config()["beta1"], get_config()["beta2"]))

    def forward(self, x, src_mask, prefix = None, dict = None):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask, prefix + ".sub1.sublayer", dict),prefix + "sub1",dict)
        if(prefix != None) :
            dict[prefix + ".output1"] = x.detach().numpy()
        x = self.residual_connections[1](x, self.feed_forward_block,prefix + "sub2",dict)
        return x
    
    def getParam(self, prefix, dict) :
        self.residual_connections[0].getParam(prefix + ".sub1", dict, self.self_attention_block)
        self.residual_connections[1].getParam(prefix + ".sub2", dict, self.feed_forward_block)

    def getForward(self, prefix, dict) :
        batch_size = get_config()["batch_size"]
        seq = get_config()["seq"]
        d_model = self.self_attention_block.d_model

        x = torch.randn(batch_size, seq, d_model)
        num_tokens = random.randint(1, seq)
        padding_mask = torch.zeros(seq, dtype=torch.uint8)
        padding_mask[:num_tokens] = 1
        encoder_mask = padding_mask.view(1, -1) & padding_mask.view(-1, 1)
        with torch.no_grad():
            y = self.forward(x, encoder_mask, prefix, dict)

        dict[prefix + ".npd"] = np.array([num_tokens]).astype(np.float32)
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

        x = torch.randn(batch_size, seq, hidden_size)
        num_tokens = random.randint(1, seq)
        padding_mask = torch.zeros(seq, dtype=torch.uint8)
        padding_mask[:num_tokens] = 1
        encoder_mask = padding_mask.view(1, -1) & padding_mask.view(-1, 1)

        self.getOriginalParam(prefix, dict)
        
        y = self.forward(x, encoder_mask, prefix, dict)
        loss = y.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.getUpdatedParam(prefix, dict)
        
        dict[prefix + ".input"] = x.detach().numpy()
        dict[prefix + ".npd"] = np.array([num_tokens]).astype(np.float32)
        dict[prefix + ".output"] = y.detach().numpy()
        dict[prefix + ".loss"] = loss.item()


def save_numpy_dict(filename: str, np_dict: dict):
    np.savez(filename, **np_dict)

encoder_self_attention_block = MultiHeadAttentionBlock(get_config()["d_model"], get_config()["head"], 0.1)
feed_forward_block = FeedForwardBlock(get_config()["d_model"], get_config()["d_ff"], 0.1)
model = EncoderBlock(get_config()["d_model"], encoder_self_attention_block, feed_forward_block, 0.1)

paramDict = {}
forwardDict = {}
backwardDict = {}

model.getParam("encoderBlock", paramDict)
save_numpy_dict("encoder_block_param.npz",paramDict)


for i in range(5) :
    model.getForward("encoderBlock", forwardDict)
    save_numpy_dict(f"encoder_block_forward{i}.npz", forwardDict)

for i in range(5):
    model.getBackward("encoderBlock", backwardDict)
    save_numpy_dict(f"encoder_block_backward{i}.npz", backwardDict)