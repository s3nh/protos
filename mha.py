import torch
import torch.nn as nn
import math
from typing import Optional, List


class Config:
    dim: int = 512

class PrepForMHA:

    def __init__(self, d_model: int, 
                 d_k: int, 
                 heads: int, 
                 bias: bool):
        super().__init__()
        self.linear = nn.Linear(d_model, heads* d_k)
        self.heads = heads
        self.dk = d_k

    def forward(self, x: torch.Tensor):
        h_shape: x.shape[-1]
        x  = self.linear(x)
        x = x.view(*h_shape, self.heads, self.d_k)
