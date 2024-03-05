from src.transformer import EncoderLayer

def test_valid_construction(encoder_layer):
    assert isinstance(
        encoder_layer, EncoderLayer
    ), "SelfAttention instance is not created properly."