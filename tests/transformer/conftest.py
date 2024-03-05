import pytest
import torch
from src.transformer.self_attention import SelfAttention

@pytest.fixture
def selfattention_params():
    return {"embed_dim": 128, "heads": 8}


@pytest.fixture
def valid_qkv_tensors(selfattention_params):
    batch_size = 3
    seq_length = 10
    tensor_shape = (batch_size, seq_length, selfattention_params["embed_dim"])
    
    # Create dummy tensors for values, keys, queries, and an optional mask
    values = torch.rand(tensor_shape)
    keys = torch.rand(tensor_shape)
    queries = torch.rand(tensor_shape)

    mask = torch.where(
        torch.rand(batch_size, selfattention_params["heads"], seq_length, seq_length) > 0.5,
        torch.ones(1),
        torch.zeros(1),
    )

    return values, keys, queries, mask

@pytest.fixture
def self_attention(selfattention_params):
    return SelfAttention(**selfattention_params)
