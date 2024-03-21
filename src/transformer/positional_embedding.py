import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, seqLen: int, dmodel: int, dropout: float = 0.1):
        super().__init__()

        self.seqLen = seqLen
        self.dmodel = dmodel
        self.dropout = nn.Dropout(dropout)

        # Shape : (Batch, Position, Dim)
        # We can expand along batch, so focus on Position & Dim for now
        pe = torch.zeros(self.seqLen, self.dmodel)

        pos = torch.arange(0, self.seqLen, dtype=torch.float).unsqueeze(
            1
        )  # (seqLen, _)

        div_term = 1000 ** (
            torch.arange(0, self.dmodel, 2, dtype=torch.float).unsqueeze(0)
            / self.dmodel
        )

        pe[:, 0::2] = torch.sin(pos / div_term)
        pe[:, 1::2] = torch.cos(pos / div_term)

        pe = pe.unsqueeze(0)  # (_, seqLen, dModel)

        # Register as buffer - They will not be trained, but exist/persist in state_dict
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        return self.dropout(x + (self.pe[:, : x.shape[1], :]).requires_grad_(False))


if __name__ == "__main__":
    a = PositionalEncoding(12, 32)
    print(a(torch.rand(12, 32)))
