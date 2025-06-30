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
from Encoder import Encoder
from Decoder import Decoder
from EncoderBlock import EncoderBlock
from DecoderBlock import DecoderBlock
from PositionalEncoding import PositionalEncoding
from InputEmbeddings import InputEmbeddings
import random


class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq, d_model) --> (batch, seq, vocab_size)
        return self.proj(x)

    def getParam(self, prefix, dict) :
        dict[prefix + ".weight"] = self.proj.weight.detach().numpy().T.copy()
        dict[prefix + ".bias"] = self.proj.bias.detach().numpy()

    def getOriginalParam(self, prefix, dict) :
        dict[prefix + ".original_weight"] = self.proj.weight.detach().numpy().T.copy()
        dict[prefix + ".original_bias"] = self.proj.bias.detach().numpy()

    def getUpdatedParam(self, prefix, dict) :
        dict[prefix + ".updated_weight"] = self.proj.weight.detach().numpy().T.copy()
        dict[prefix + ".updated_bias"] = self.proj.bias.detach().numpy()

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=get_config()["lr"], betas=(get_config()["beta1"], get_config()["beta2"]))

    def encode(self, src, src_mask, prefix=None, dict=None):
        # (batch, seq, d_model)
        x1 = self.src_embed(src)
        x2 = self.src_pos(x1)
        if(prefix != None) :
            dict[prefix + ".embedOut"] = x1.detach().numpy()
            dict[prefix + ".posOut"] = x2.detach().numpy()
        return self.encoder(x2, src_mask, prefix, dict)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def getParam(self, prefix, dict) :
        self.encoder.getParam(prefix + ".encoder", dict)
        self.decoder.getParam(prefix + ".decoder", dict)
        self.src_embed.getParam(prefix + ".src_embed", dict)
        self.tgt_embed.getParam(prefix + ".tgt_embed", dict)
        self.projection_layer.getParam(prefix + ".projection_layer", dict)

    def getForward(self, prefix, dict) :
        batch_size = get_config()["batch_size"]
        seq = get_config()["seq"]

        xs = torch.randint(0, self.src_embed.vocab_size, (batch_size, seq))
        xt = torch.randint(0, self.tgt_embed.vocab_size, (batch_size, seq))
        num_src_tokens = random.randint(1, seq)
        num_tgt_tokens = random.randint(1, seq)
        src_padding_mask = torch.zeros(seq, dtype=torch.uint8)
        src_padding_mask[:num_src_tokens] = 1
        tgt_padding_mask = torch.zeros(seq, dtype=torch.uint8)
        tgt_padding_mask[:num_tgt_tokens] = 1
        padding_mask = torch.zeros(seq, dtype=torch.uint8)
        padding_mask[:num_src_tokens] = 1
        lookahead_mask = torch.tril(torch.ones((seq, seq), dtype=torch.uint8))
        tgt_mask = lookahead_mask & tgt_padding_mask.view(1, -1) & tgt_padding_mask.view(-1, 1)
        src_mask = src_padding_mask.view(1, -1)
        encoder_mask = padding_mask.view(1, -1) & padding_mask.view(-1, 1)

        with torch.no_grad():
            encoder_output = self.encode(xs, encoder_mask, prefix, dict)
            decoder_output = self.decode(encoder_output, src_mask, xt, tgt_mask)
            y = self.project(decoder_output)

        dict[prefix + ".npd"] = np.array([num_src_tokens, num_tgt_tokens]).astype(np.float32)
        dict[prefix + ".input1"] = xs.detach().numpy().astype(np.float32)
        dict[prefix + ".input2"] = xt.detach().numpy().astype(np.float32)
        dict[prefix + ".outpute"] = encoder_output.detach().numpy()
        dict[prefix + ".outputd"] = decoder_output.detach().numpy()
        dict[prefix + ".output"] = y.detach().numpy()

    def getOriginalParam(self, prefix, dict) :
        self.encoder.getOriginalParam(prefix + ".encoder", dict)
        self.decoder.getOriginalParam(prefix + ".decoder", dict)
        self.src_embed.getOriginalParam(prefix + ".src_embed", dict)
        self.tgt_embed.getOriginalParam(prefix + ".tgt_embed", dict)
        self.projection_layer.getOriginalParam(prefix + ".projection_layer", dict)

    
    def getUpdatedParam(self, prefix, dict) :
        self.encoder.getUpdatedParam(prefix + ".encoder", dict)
        self.decoder.getUpdatedParam(prefix + ".decoder", dict)
        self.src_embed.getUpdatedParam(prefix + ".src_embed", dict)
        self.tgt_embed.getUpdatedParam(prefix + ".tgt_embed", dict)
        self.projection_layer.getUpdatedParam(prefix + ".projection_layer", dict)

    def getBackward(self, prefix, dict) :
        batch_size = get_config()["batch_size"]
        seq = get_config()["seq"]

        xs = torch.randint(0, self.src_embed.vocab_size, (batch_size, seq))
        xt = torch.randint(0, self.tgt_embed.vocab_size, (batch_size, seq))
        num_src_tokens = random.randint(1, seq)
        num_tgt_tokens = random.randint(1, seq)
        src_padding_mask = torch.zeros(seq, dtype=torch.uint8)
        src_padding_mask[:num_src_tokens] = 1
        tgt_padding_mask = torch.zeros(seq, dtype=torch.uint8)
        tgt_padding_mask[:num_tgt_tokens] = 1
        padding_mask = torch.zeros(seq, dtype=torch.uint8)
        padding_mask[:num_src_tokens] = 1
        lookahead_mask = torch.tril(torch.ones((seq, seq), dtype=torch.uint8))
        tgt_mask = lookahead_mask & tgt_padding_mask.view(1, -1) & tgt_padding_mask.view(-1, 1)
        src_mask = src_padding_mask.view(1, -1)
        encoder_mask = padding_mask.view(1, -1) & padding_mask.view(-1, 1)

        self.getOriginalParam(prefix, dict)
        
        encoder_output = self.encode(xs, encoder_mask)
        decoder_output = self.decode(encoder_output, src_mask, xt, tgt_mask)
        y = self.project(decoder_output)
        loss = y.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.getUpdatedParam(prefix, dict)
        
        dict[prefix + ".npd"] = np.array([num_src_tokens, num_tgt_tokens]).astype(np.float32)
        dict[prefix + ".input1"] = xs.detach().numpy().astype(np.float32)
        dict[prefix + ".input2"] = xt.detach().numpy().astype(np.float32)
        dict[prefix + ".outpute"] = encoder_output.detach().numpy()
        dict[prefix + ".output"] = y.detach().numpy()
        dict[prefix + ".loss"] = loss.item()
    
    def project(self, x):
        # (batch, seq, vocab_size)
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq: int, tgt_seq: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer

N = 6
def save_numpy_dict(filename: str, np_dict: dict):
    np.savez(filename, **np_dict)

model = build_transformer(128, 128, get_config()["seq"],  get_config()["seq"],  get_config()["d_model"], N,  get_config()["head"],  0.1,  get_config()["d_ff"])

paramDict = {}
forwardDict = {}
backwardDict = {}

model.getParam("transformer", paramDict)
save_numpy_dict("transformer_param.npz",paramDict)

for i in range(5) :
    model.getForward("transformer", forwardDict)
    save_numpy_dict(f"transformer_forward{i}.npz", forwardDict)

for i in range(5):
    model.getBackward("transformer", backwardDict)
    save_numpy_dict(f"transformer_backward{i}.npz", backwardDict)

