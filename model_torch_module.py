import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Optional
    

    
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
    


class CoodebookRecontructionLayer(nn.Module):

    def __init__(self, d_model:int, vocab_size:int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)
    

    
class Transformer(nn.Transformer):
    def __init__(self, 
                 codebook_size: int , 
                 codebook_num: int ,
                 max_len_token: int,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 padding_token: Optional[int] = 1025,
                 d_model: int= 128,
                 nhead: int= 8,
                 dropout: float=0.1,
                 d_ff: int=2048,
                 batch_first: bool = True) -> None:
        super().__init__(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, 
                         dim_feedforward=d_ff, dropout=dropout, batch_first=batch_first)
        
        self.d_model = d_model
        self.codebook_num = codebook_num
        self.codebook_size = codebook_size
        self.embed = nn.ModuleList([nn.Embedding(codebook_size, d_model, padding_token) for _ in range(codebook_num)])
        self.pos = PositionalEncoding(d_model, max_len_token, dropout)
        self.projection_layer = nn.ModuleList([CoodebookRecontructionLayer(d_model, codebook_size) for _ in range(codebook_num)])

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor = None, tgt_mask: torch.Tensor=None, memory_mask=None, 
                src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        
        K = src.size(1)
        assert K == self.codebook_num, f"Expected {self.codebook_num} codebooks, got {K}"
        
        embeds_src_list = []
        embeds_tgt_list = []
        proj_list = []
        
        # Generar embeddings de src y tgt
        for i, emb in enumerate(self.embed):
            embeds_src_list.append(emb(src[:,i]).unsqueeze(1))
            embeds_tgt_list.append(emb(tgt[:,i]).unsqueeze(1))
        
        # Concatenar y sumar embeddings
        src = torch.cat(embeds_src_list, dim=1).sum(dim=1)
        tgt = torch.cat(embeds_tgt_list, dim=1).sum(dim=1)
        
        # Eliminar las variables temporales
        del embeds_src_list, embeds_tgt_list
        
        # Aplicar la función de posicionamiento
        src = self.pos(src)
        tgt = self.pos(tgt)
        
        # Pasar por la red de transformación (super)
        out = super().forward(src, tgt, src_mask, tgt_mask, memory_mask,
                            src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)
        
        # Generar proyecciones
        for i, proj_layer in enumerate(self.projection_layer):
            proj_list.append(proj_layer(out).unsqueeze(1))
        
        # Concatenar las proyecciones y aplicar softmax
        proj = torch.cat(proj_list, dim=1)
        return F.log_softmax(proj, dim=-1)
        
def test():
    model = Transformer(codebook_size=1026, codebook_num=8, max_len_token=100)
    src = torch.randint(0, 1025, (4, 8, 100))
    tgt = torch.randint(0, 1025, (4, 8, 100))
    src_mask = torch.ones(4, 100, dtype=torch.bool)
    tgt_mask = torch.ones(4, 100, dtype=torch.bool)
    mask = model.generate_square_subsequent_mask(100)
    out = model(src, tgt, tgt_mask=mask, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask)
    print(f"Output: {out.shape}")
    print("-----------------------------------")
    

if __name__ == "__main__":
    test()