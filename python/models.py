from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch 
from torch import nn

class ElementwiseModel(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.relu((x + y) * 0.5) 

class TinyMLP(nn.Module):
    def __int__(self, width: int=16) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.fc1 = nn.Linear(width, width*2)
        self.fc2 = nn.Linear(width*2, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))

class MatMulRelu(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.relu(a @ b)

class TinyConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.conv(x))

