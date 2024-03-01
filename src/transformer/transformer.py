import torch


class SelfAttention(torch.nn.Module):
    def __init__(self, embed_size: int, heads: int):
        """Implement a Multi-Headed self attention block

        :param embed_size: Input dimensions
        :type embed_size: int
        :param heads: Number of heads
        :type heads: int
        """
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = self.embed_size // self.heads

        assert (
            self.embed_size % self.head_dim == 0
        ), "Embed size must be evenly divisible by heads"


        self.values_proj = torch.nn.Linear(self.embed_size, self.embed_size)
        self.keys_proj = torch.nn.Linear(self.embed_size, self.embed_size)
        self.queries_proj = torch.nn.Linear(self.embed_size, self.embed_size)

    def forward(self, values: torch.Tensor, keys: torch.Tensor, queries: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # validate dimensions:
        # 1. All values must use the same embedding size
        # 2. The sequence length for values and keys 
        assert (queries.shape[-1]==keys.shape[-1]), "queries and keys must have same query (embed) size"
        assert (values.shape[-2]==keys.shape[-2]), "values and keys must have same sequence length"
        
        # Get input shapes
        B = queries.shape[0] # Size of batch
        q_len, k_len, v_len = queries.shape[1], keys.shape[1], values.shape[1] # Sequence length for each

        # Project to the Q, K, & V
        queries = self.queries_proj(queries)
        keys = self.keys_proj(keys)
        values = self.values_proj(values)

        # Reshape
        queries = queries.reshape(B, q_len, self.heads, self.head_dim) 
        keys = keys.reshape(B, k_len, self.heads, self.head_dim)
        values = values.reshape(B, v_len, self.heads, self.head_dim)

        # torch.einsum convenient way to perform bmm between query and key
        # across each sample and each head
        energy = torch.einsum('bqhd,bkhd->bhqk',queries,keys) # Output shape (B, heads, q_len, k_len)
        energy = energy / (q_len ** (1.0/2.0))
        
        if mask:
            energy = energy.masked_fill(mask==0, float("1e-20"))

        energy = torch.softmax(energy, 3)

        # Perform bmm between scaled energy and value (performs the reshapes)
        scaledAttention = torch.einsum('bhqk,bvhd->bqhd',energy,values) # Output shape (B, q_len, heads, head_dim)

        return scaledAttention
    

