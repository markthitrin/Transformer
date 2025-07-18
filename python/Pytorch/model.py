import torch
import torch.nn as nn
import math
import time

class Timer:
    def __init__(self):
        self.t = []
        self.round = -1
        self.i = 0
        self.prev_time = 0.0

    def restart(self):
        self.i = 0
        self.round += 1
        self.prev_time = time.perf_counter()

    def checkpoint(self):
        now = time.perf_counter()
        elapsed = now - self.prev_time
        self.prev_time = now

        if self.i >= len(self.t):
            self.t.append([])

        if len(self.t[self.i]) <= self.round:
            self.t[self.i].append(elapsed)
        else:
            self.t[self.i][self.round] = elapsed

        self.i += 1
        return elapsed

    def get_time(self):
        result = []
        for i, times in enumerate(self.t):
            if not times:
                result.append(None)
                continue
            sorted_times = sorted(times)
            n = len(sorted_times)
            start = min(int(n * 0.1), 20)
            end = max((n * 9 + 9) // 10, n - 20)
            trimmed = sorted_times[start:end]
            if trimmed:
                avg = sum(trimmed) / len(trimmed)
                result.append(avg)
            else:
                result.append(None)
        return result
    
    def get_time_std(self):
        result = []
        for i, times in enumerate(self.t):
            if not times:
                result.append(None)
                continue
            sorted_times = sorted(times)
            n = len(sorted_times)
            start = min(int(n * 0.1), 20)
            end = max((n * 9 + 9) // 10, n - 20)
            trimmed = sorted_times[start:end]
            if trimmed:
                avg = sum(trimmed) / len(trimmed)
                variance = sum((x - avg) ** 2 for x in trimmed) / len(trimmed)
                std = math.sqrt(variance)
                result.append(std)
            else:
                result.append(None)
        return result

timer = Timer()

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        y = self.embedding(x) * math.sqrt(self.d_model)
        timer.checkpoint()
        
        return y
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq = seq
        self.dropout = nn.Dropout(dropout)
        
        # Create a matrix of shape (seq, d_model)
        pe = torch.zeros(seq, d_model)
        
        # Create a vector of shape (seq)
        position = torch.arange(0, seq, dtype=torch.float).unsqueeze(1) # (seq, 1)
        
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq, d_model)
        
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq, d_model)
        timer.checkpoint()
        
        y = self.dropout(x)
        timer.checkpoint()
        
        return y

class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq, d_model) --> (batch, seq, vocab_size)
        y = self.proj(x)
        timer.checkpoint()
        
        return y
    
class Transformer(nn.Module):

    def __init__(self, transformer: nn.Transformer, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        self.transformer = transformer

    def forward(self, src: torch.Tensor,tgt: torch.Tensor, src_key_padding_mask: torch.Tensor, tgt_key_padding_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        src_emb = self.src_pos(self.src_embed(src))
        tgt_emb = self.tgt_pos(self.tgt_embed(tgt))

        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_mask=tgt_mask,
        )
        timer.checkpoint()

        return self.projection_layer(output)

    
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq: int, tgt_seq: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=256) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq, dropout)
    
    transformer = nn.Transformer(
        d_model=d_model,
        nhead=h,
        num_encoder_layers=N,
        num_decoder_layers=N,
        dim_feedforward=d_ff,
        dropout=dropout,
        layer_norm_eps=1e-05,
        batch_first=True,
        norm_first=True,
        bias=True,
        dtype=torch.float32
    )
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(transformer, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer