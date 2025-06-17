import torch
import torch.nn as nn
import math
import numpy as np
from config import get_config
from LayerNormalization import LayerNormalization
from MultiheadAttention import MultiHeadAttentionBlock
from FeedForwardBlock import FeedForwardBlock
from DecoderBlock import DecoderBlock
import random

class ManualDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.scale = 1.0 - dropout

    def forward(self, x):
        return x / self.scale

class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=get_config()["lr"], betas=(get_config()["beta1"], get_config()["beta2"]))

    def forward(self, x, encoder_output, src_mask, tgt_mask, prefix=None, dict=None):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        if(prefix != None) :
            dict[prefix + ".outputsub"] = x.detach().numpy()
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
        xe = torch.randn(batch_size, seq, d_model)
        num_src_tokens = random.randint(1, seq)
        num_tgt_tokens = random.randint(1, seq)
        src_padding_mask = torch.zeros(seq, dtype=torch.uint8)
        src_padding_mask[:num_src_tokens] = 1
        tgt_padding_mask = torch.zeros(seq, dtype=torch.uint8)
        tgt_padding_mask[:num_tgt_tokens] = 1
        lookahead_mask = torch.tril(torch.ones((seq, seq), dtype=torch.uint8))
        tgt_mask = lookahead_mask & tgt_padding_mask.view(1, -1) & tgt_padding_mask.view(-1, 1)
        src_mask = src_padding_mask.view(1, -1)

        with torch.no_grad():
            y = self.forward(x, xe, src_mask, tgt_mask, prefix, dict)

        dict[prefix + ".npd"] = np.array([num_src_tokens, num_tgt_tokens]).astype(np.float32)
        dict[prefix + ".input1"] = x.detach().numpy()
        dict[prefix + ".input2"] = xe.detach().numpy()
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
        d_model = self.norm.alpha.shape[0]

        x = torch.randn(batch_size, seq, d_model)
        xe = torch.randn(batch_size, seq, d_model)
        num_src_tokens = random.randint(1, seq)
        num_tgt_tokens = random.randint(1, seq)
        src_padding_mask = torch.zeros(seq, dtype=torch.uint8)
        src_padding_mask[:num_src_tokens] = 1
        tgt_padding_mask = torch.zeros(seq, dtype=torch.uint8)
        tgt_padding_mask[:num_tgt_tokens] = 1
        lookahead_mask = torch.tril(torch.ones((seq, seq), dtype=torch.uint8))
        tgt_mask = lookahead_mask & tgt_padding_mask.view(1, -1) & tgt_padding_mask.view(-1, 1)
        src_mask = src_padding_mask.view(1, -1)

        self.getOriginalParam(prefix, dict)
        
        y = self.forward(x, xe, src_mask, tgt_mask)
        loss = y.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.getUpdatedParam(prefix, dict)
        
        dict[prefix + ".npd"] = np.array([num_src_tokens, num_tgt_tokens]).astype(np.float32)
        dict[prefix + ".input1"] = x.detach().numpy()
        dict[prefix + ".input2"] = xe.detach().numpy()
        dict[prefix + ".output"] = y.detach().numpy()
        dict[prefix + ".loss"] = loss.item()
    
def save_numpy_dict(filename: str, np_dict: dict):
    np.savez(filename, **np_dict)

N = 6

decoder_blocks = []
for _ in range(N):
    decoder_self_attention_block = MultiHeadAttentionBlock(get_config()["d_model"], 8, 0.1)
    decoder_cross_attention_block = MultiHeadAttentionBlock(get_config()["d_model"], 8, 0.1)
    feed_forward_block = FeedForwardBlock(get_config()["d_model"], get_config()["d_ff"], 0.1)
    decoder_block = DecoderBlock(get_config()["d_model"], decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, 0.1)
    decoder_blocks.append(decoder_block)

model = Decoder(get_config()["d_model"], nn.ModuleList(decoder_blocks))

paramDict = {}
forwardDict = {}
backwardDict = {}

model.getParam("decoder", paramDict)
save_numpy_dict("decoder_param.npz",paramDict)

for i in range(5) :
    model.getForward("decoder", forwardDict)
    save_numpy_dict(f"decoder_forward{i}.npz", forwardDict)

for i in range(5):
    model.getBackward("decoder", backwardDict)
    save_numpy_dict(f"decoder_backward{i}.npz", backwardDict)