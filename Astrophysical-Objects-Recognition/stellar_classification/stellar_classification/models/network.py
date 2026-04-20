"""PyTorch neural network for stellar classification."""

import torch.nn as nn


class SimpleNN(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
