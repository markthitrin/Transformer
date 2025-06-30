import torch
import torch.nn as nn
import math
import numpy as np
from config import get_config
from EncoderBlock import EncoderBlock
from LayerNormalization import LayerNormalization
from LayerNormalization import LayerNormalization
from MultiheadAttention import MultiHeadAttentionBlock
from FeedForwardBlock import FeedForwardBlock
import random

class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=get_config()["lr"], betas=(get_config()["beta1"], get_config()["beta2"]))

    def forward(self, x, mask, prefix=None, dict=None):
        i = 0
        for layer in self.layers:
            if(prefix != None) :
                x = layer(x, mask, prefix + f".layer{i}", dict)
            else :
                x = layer(x, mask)
            if(prefix != None) :
                dict[prefix + f".encoderAfterlayer{i}"] = x.detach().numpy()
            i += 1
        return self.norm(x)
    
    def getParam(self, prefix, dict) :
        for i in range(len(self.layers)) :
            self.layers[i].getParam(prefix + f".layer{i}", dict)
        self.norm.getParam(prefix + ".norm", dict)

    def getForward(self, prefix, dict) :
        batch_size = get_config()["batch_size"]
        seq = get_config()["seq"]
        d_model = self.norm.alpha.shape[0]

        x = torch.randn(batch_size, seq, d_model)
        num_tokens = random.randint(1, seq)
        padding_mask = torch.zeros(seq, dtype=torch.uint8)
        padding_mask[:num_tokens] = 1
        encoder_mask = padding_mask.view(1, -1) & padding_mask.view(-1, 1)
        with torch.no_grad():
            y = self.forward(x, encoder_mask)

        dict[prefix + ".npd"] = np.array([num_tokens]).astype(np.float32)
        dict[prefix + ".input"] = x.detach().numpy()
        dict[prefix + ".output"] = y.detach().numpy()

    def getOriginalParam(self, prefix, dict) :
        for i in range(len(self.layers)) :
            self.layers[i].getOriginalParam(prefix + f".layer{i}", dict)
        self.norm.getOriginalParam(prefix + ".norm", dict)
    
    def getUpdatedParam(self, prefix, dict) :
        for i in range(len(self.layers)) :
            self.layers[i].getUpdatedParam(prefix + f".layer{i}", dict)
        self.norm.getUpdatedParam(prefix + ".norm", dict)

    def getBackward(self, prefix, dict) :
        batch_size = get_config()["batch_size"]
        seq = get_config()["seq"]
        hidden_size = self.norm.alpha.shape[0]

        
        x = torch.randn(batch_size, seq, hidden_size)
        num_tokens = random.randint(1, seq)
        padding_mask = torch.zeros(seq, dtype=torch.uint8)
        padding_mask[:num_tokens] = 1
        encoder_mask = padding_mask.view(1, -1) & padding_mask.view(-1, 1)
        self.getOriginalParam(prefix, dict)
        
        y = self.forward(x, encoder_mask)
        loss = y.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.getUpdatedParam(prefix, dict)
        
        dict[prefix + ".npd"] = np.array([num_tokens]).astype(np.float32)
        dict[prefix + ".input"] = x.detach().numpy()
        dict[prefix + ".output"] = y.detach().numpy()
        dict[prefix + ".loss"] = loss.item()