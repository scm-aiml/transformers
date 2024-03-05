import torch
from torch import nn
from src.transformer import SelfAttention
from typing import Optional

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, heads: int, forwardExpand: int, dropout: float):
        super(EncoderLayer, self).__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        self.forwardExpand = forwardExpand
        
        # First sub-layer
        self.attn = SelfAttention(heads=self.heads, embed_dim=self.embed_dim)
        self.layernorm1 = nn.LayerNorm(self.embed_dim)

        # Second Layer
        self.feed_forwad = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * self.forwardExpand),
            nn.ReLU(),
            nn.Linear(self.embed_dim * self.forwardExpand, self.embed_dim),
        )
        self.layernorm2 = nn.LayerNorm(self.embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        attention = self.attn(x, x, x, mask)
        x = self.layernorm1(self.dropout(attention) + x)
        
        forward_output = self.feed_forwad(x) 
        x = self.layernorm2(x + self.dropout(forward_output))

        return x
