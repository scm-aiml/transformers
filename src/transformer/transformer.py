import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, heads: int, bias: bool = False):
        """Implement a Multi-Headed self attention block

        :param embed_dim: Input dimensions
        :type embed_dim: int
        :param heads: Number of heads
        :type heads: int
        """
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = self.embed_dim // self.heads

        assert (
            self.embed_dim % self.head_dim == 0
        ), "Embed size must be evenly divisible by heads"

        self.values_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.keys_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.queries_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.fc_out = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        values: torch.Tensor,
        keys: torch.Tensor,
        queries: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """_summary_

        :param values: Values tensor
        :type values: torch.Tensor
        :param keys: Keys tensor
        :type keys: torch.Tensor
        :param queries: Queries tensor
        :type queries: torch.Tensor
        :param mask: Optional Mask, defaults to None
        :type mask: torch.Tensor, optional
        :raises ValueError: Embed dimension mismatch (queries and keys)
        :raises ValueError: Sequence lenght mistach (keys and values)
        :return: Multi-Headed SelfAttention values
        :rtype: torch.Tensor
        """
        # validate dimensions:
        # 1) Q x K.t -> energy (q_seq_len, k_seq_len)
        if queries.shape[-1] != keys.shape[-1]:
            raise ValueError(
                f"queries and keys embed dimension mismatch.\nqueries: {queries.shape[-1]}\nkeys: {keys.shape[-1]}\n"
            )

        # 2) energy x V -> (q_seq_len, v_embed_dim)
        if keys.shape[-2] != values.shape[-2]:
            raise ValueError(
                f"keys and values sequence length mismatch.\nkeys: {keys.shape[-2]}\nvalues: {values.shape[-2]}\n"
            )

        # Get input shapes
        B = queries.shape[0]  # Batch size
        k_len = torch.tensor(keys.shape[1])

        # Project
        keys = self.keys_proj(keys)
        values = self.values_proj(values)
        queries = self.queries_proj(queries)

        # Reshape
        values = values.reshape(B, -1, self.heads, self.head_dim)
        keys = keys.reshape(B, -1, self.heads, self.head_dim)
        queries = queries.reshape(B, -1, self.heads, self.head_dim)

        # Perform multi-head attention
        energy = torch.einsum(
            "bqhd,bkhd->bhqk", queries, keys
        )  # Output shape (B, heads, q_len, k_len)

        energy = energy / torch.sqrt(k_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("1e-20"))

        energy = torch.softmax(energy, 3)

        attention = torch.einsum(
            "bhqk,bvhd->bqhd", energy, values
        )  # Output shape (B, q_len, heads, head_dim)

        attention = attention.reshape(B, -1, self.embed_dim)

        return self.fc_out(attention)
