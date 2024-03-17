import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, seqLen: int, dmodel: int):
        self.seqLen = seqLen
        self.dmodel = dmodel

        # Shape : (Batch, Position, Dim)
        # We can expand along batch, so focus on Position & Dim for now

        pos = torch.arange(0, self.seqLen, dtype=torch.float).unsqueeze(1)
        dim = torch.arange(0, self.dmodel, dtype=torch.float).unsqueeze(0)

        pe = pos / (10000 ** (2*dim/self.dmodel)) 
        print(pe)


if __name__ == "__main__":
    a = PositionalEncoding(4, 3)

