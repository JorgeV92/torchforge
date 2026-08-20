from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch 
from torch import nn

class ElementwiseModel(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.relu((x + y) * 0.5) 

class TinyMLP(nn.Module):
    def __init__(self, width: int = 16) -> None:
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

@dataclass(frozen=True)
class ModelSpec:
    name: str
    description: str
    factory: Callable[[], nn.Module]
    inputs: Callable[[], tuple[torch.Tensor, ...]]

def _elementwise_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(1)
    return (torch.randn(4, 8), torch.randn(4, 8))

def _mlp_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(1)
    return (torch.randn(4, 16), )

def _matmul_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(1)
    return (torch.randn(8, 16), torch.randn(16, 12))

def _conv_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(1)
    return (torch.randn(1, 3, 16, 16), )

MODEL_SPECS: dict[str, ModelSpec] = {
    "elementwise": ModelSpec(
        "elementwise", "add -> scale -> relu", ElementwiseModel, _elementwise_inputs
    ),
    "mlp": ModelSpec("mlp", "two-layer MLP with ReLU", TinyMLP, _mlp_inputs),
    "matmul": ModelSpec("matmul", "matrix multiplication followed by ReLU", MatMulRelu, _matmul_inputs),
    "conv": ModelSpec("conv", "small 2D convolution followed by ReLU", TinyConv, _conv_inputs),
}

def model_names() -> list[str]:
    return sorted(MODEL_SPECS)

def get_model_spec(name: str) -> ModelSpec:
    try:
        return MODEL_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"unknown model {name!r}; choose from {', '.join(model_names())}") from exc
