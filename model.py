import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

import yaml
from typing import Optional

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x)))) 
    


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int, padding_token: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model, padding_token)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)
    

    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)
    


class ResidualConnection(nn.Module):
    
        def __init__(self, features: int, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(features)
    
        def forward(self, x, sublayer):
            return x + self.dropout(sublayer(self.norm(x)))
        


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float, flash: bool) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"
        
        self.flash = flash # Enable flash attention
        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)
        self.dropoutvalue = dropout

    @staticmethod
    def attention(query, key, value, mask, dropout, flash_attention, is_casual):
        if flash_attention:
            return F.scaled_dot_product_attention(query, key, value, mask, dropout, is_causal=is_casual), None
        
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores
        
    def forward(self, q, k, v, mask, is_casual: bool = False):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, 
                                                                     self.dropoutvalue if self.flash else self.dropout, 
                                                                     self.flash, is_casual)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)



class EncoderBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int=2048,  nhead: int=8, dropout: float = 0.1, flash: bool = True) -> None:
        super().__init__()
        self.self_attention_block = MultiHeadAttentionBlock(d_model, nhead, dropout, flash)
        self.feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)])
    
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


    
class Encoder(nn.Module):

    def __init__(self, d_model: int, layer: EncoderBlock, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    


class DecoderBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int=2048,  nhead: int=8, dropout: float = 0.1, flash: bool = True) -> None:
        super().__init__()
        self.self_attention_block = MultiHeadAttentionBlock(d_model, nhead, dropout, flash)
        self.cross_attention_block = MultiHeadAttentionBlock(d_model, nhead, dropout, flash)
        self.feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask, is_casual=True))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    


class Decoder(nn.Module):

    def __init__(self, d_model: int, layer: DecoderBlock, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    


class CoodebookRecontructionLayer(nn.Module):

    def __init__(self, d_model:int, vocab_size:int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)
    

    
class Transformer(nn.Module):
    """
    The Transformer model with the addition of the codebook reconstruction 
    layer and the codebook projection layer to the model architecture
    
    :params codebook_size: int, the size of the codebook
    :params codebook_num: int, the number of codebooks
    :params max_len_token: int, the maximum length of the token
    :params num_encoder_layers: int, the number of encoder layers
    :params num_decoder_layers: int, the number of decoder layers
    :params pad_token: Optional[int], the padding token
    :params d_model: int, the dimension of the model
    :params nhead: int, the number of heads in the multihead attention
    :params dropout: float, the dropout rate
    :params d_ff: int, the dimension of the feed forward network
    :params flash: bool, whether to use the flash attention mechanism
    """
    def __init__(self, codebook_size: int, codebook_num: int, max_len_token: int, max_len_midi: int,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6, pad_token: Optional[int] = None,
                 pad_midi: Optional[int] = None, midi_size: int = 30000, d_model: int= 128, nhead: int=8,
                 dropout: float=0.1, d_ff: int=2048, flash: bool = True
                 ) -> None:
        super().__init__()
        
        self._d_model = d_model
        self._codebook_num = codebook_num
        self._codebook_size = codebook_size
        
        self.embed = nn.ModuleList([InputEmbeddings(d_model, codebook_size, pad_token) for _ in range(codebook_num)])
        self.midi_embed = InputEmbeddings(d_model, midi_size, pad_midi)
        
        self.pos = PositionalEncoding(d_model, max_len_token, dropout)
        self.midi_pos = PositionalEncoding(d_model, max_len_midi, dropout)
        
        self.encoder_block = EncoderBlock(d_model, d_ff, nhead , dropout, flash)
        self.encoder = Encoder(d_model, self.encoder_block, num_encoder_layers)
        
        self.decoder_block = DecoderBlock(d_model, d_ff, nhead , dropout, flash)
        self.decoder = Decoder(d_model, self.decoder_block, num_decoder_layers)

        self.projection_layer = nn.ModuleList([CoodebookRecontructionLayer(d_model, codebook_size) for _ in range(codebook_num)])

        self.apply(self._init_weights)
        
    @property
    def d_model(self):
        return self._d_model
    
    @property
    def codebook_num(self):
        return self._codebook_num
    
    @property
    def codebook_size(self):
        return self._codebook_size
    
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    # Encoder and Decoder forward pass
    # the forward pass is similar to the original transformer model with the addition of the codebook
    # The codebook is a tensor of shape (batch, codebook_num, seq_len, codebook_size)
    
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None):
        # (batch, seq_len, d_model)
        
        src = self.midi_embed(src)
        src = self.pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: Optional[torch.Tensor] = None, tgt_mask: Optional[torch.Tensor] = None):
        # (batch, seq_len, d_model)
        B, K, T = tgt.shape

        assert K == self.codebook_num, f"Expected {self.codebook_num} codebooks, got {K}"

        embeds = torch.empty((B, K, T, self._d_model), device=DEVICE)
        for i, emb in enumerate(self.embed):
            embeds[:,i] = emb(tgt[:,i])
        embeds = embeds.sum(dim=1)
        tgt = self.pos(embeds)
        return self.decoder(tgt, src, src_mask, tgt_mask)
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        proj = torch.empty(x.shape[0], self.codebook_num, x.shape[1], self.codebook_size).to(x.device)
        for i, proj_layer in enumerate(self.projection_layer):
            proj[:,i] = proj_layer(x)
        return proj

def test(config):
    torch.set_float32_matmul_precision("high")
    model = Transformer(**config["Transformer"]).to(DEVICE)
    src = torch.randint(0, 30000, (5, 11000)).to(DEVICE)
    tgt = torch.randint(0, 1025, (5, 8, 11000)).to(DEVICE)
    # src_mask = torch.ones(4, 1, 100)
    # tgt_mask = torch.ones(4, 1, 100)
    with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
        t0 = time.time()
        enc = model.encode(src)
        dec = model.decode(enc, tgt)
        proj = model.project(dec)
        t1 = time.time()
    print("-----------------------------------")
    print(f"Encoder output: {enc.shape}")
    print(f"Decoder output: {dec.shape}")
    print(f"Projection output: {proj.shape}")
    print("-----------------------------------")
    print(f"Time taken: {t1 - t0:.4f}")

if __name__ == "__main__":
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    test(config)