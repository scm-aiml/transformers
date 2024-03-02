import pytest
from src.transformer.transformer import SelfAttention

def test_valid_construction(self_attention):
    assert isinstance(
        self_attention, SelfAttention
    ), "SelfAttention instance is not created properly."


def test_forward_pass(self_attention, valid_tensors):
    values, keys, queries, _ = valid_tensors
    self_attention(values, keys, queries)

def test_dimension_mismatch(self_attention, valid_tensors):
    values, keys, queries, mask = valid_tensors
    # Modify queries to cause dimension mismatch
    queries_wrong = queries[:, :, :-1]  # Remove one dimension to cause mismatch
    with pytest.raises(ValueError):
        self_attention(values, keys, queries_wrong, mask)

    # Modify keys to cause sequence length mismatch with values
    keys_wrong = keys[:, :-1, :]  # Remove one sequence length to cause mismatch
    with pytest.raises(ValueError):
        self_attention(values, keys_wrong, queries, mask)

def test_mask(self_attention, valid_tensors):
    values, keys, queries, mask = valid_tensors
    self_attention(values, keys, queries, mask)
