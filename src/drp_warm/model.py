"""MLP regressor for cancer drug-response prediction.

Architecture matches the original 2018 Keras implementation
(`pilot1/apps/accl_trn/utils/ml_models.py::KERAS_REGRESSOR`):

    input_dim -> 1000 -> 1000 -> 500 -> 250 -> 125 -> 60 -> 30 -> 1

ReLU activations, dropout after every hidden layer except the first, and a
ReLU on the output to constrain predictions to non-negative reals (the
target is AUC ∈ [0, 1]).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


DEFAULT_HIDDEN: tuple[int, ...] = (1000, 1000, 500, 250, 125, 60, 30)


class DRPRegressor(nn.Module):
    """Fully-connected regressor over concatenated GE + DD features."""

    def __init__(
        self,
        input_dim: int,
        hidden: Sequence[int] = DEFAULT_HIDDEN,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for i, width in enumerate(hidden):
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU(inplace=True))
            if i > 0:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
