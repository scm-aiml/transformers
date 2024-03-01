from src.transformer import SelfAttention
import torch

def test_selfattention():
    embed_size = 32
    heads = 8
    head_dim = 4
    selfattn = SelfAttention(embed_size=embed_size, heads=heads)

    assert (selfattn.head_dim == head_dim), "head dimension incorrect"
    
    q = k = v =  torch.rand(5,10,embed_size)

    selfattn(q,k,v)