import pytest
import torch
from src.transformer.self_attention import SelfAttention

def test_valid_construction(self_attention):
    assert isinstance(
        self_attention, SelfAttention
    ), "SelfAttention instance is not created properly."

def test_forward_pass(self_attention, valid_qkv_tensors):
    values, keys, queries, _ = valid_qkv_tensors
    self_attention(values, keys, queries)

def test_dimension_mismatch(self_attention, valid_qkv_tensors):
    values, keys, queries, mask = valid_qkv_tensors
    
    # Queries and Keys embed dimension mismatch
    queries_wrong = queries[:, :, :-1]  
    with pytest.raises(ValueError):
        self_attention(values, keys, queries_wrong, mask)

    # Keys and Values sequence length mismatch
    keys_wrong = keys[:, :-1, :]  
    with pytest.raises(ValueError):
        self_attention(values, keys_wrong, queries, mask)

def test_mask(self_attention, valid_qkv_tensors):
    values, keys, queries, mask = valid_qkv_tensors
    out: torch.Tensor = self_attention(values, keys, queries, mask)
    assert(out.shape==queries.shape), "Output shape of SelAttention forward pass incorrect"
