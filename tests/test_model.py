"""Tests for the DRP regressor architecture."""

from __future__ import annotations

import torch

from drp_warm.model import DEFAULT_HIDDEN, DRPRegressor


def test_forward_shape():
    model = DRPRegressor(input_dim=100)
    x = torch.randn(8, 100)
    y = model(x)
    assert y.shape == (8,), f"expected (8,), got {tuple(y.shape)}"


def test_output_is_non_negative():
    # ReLU on output should clamp at zero.
    model = DRPRegressor(input_dim=20)
    x = torch.randn(64, 20)
    with torch.no_grad():
        y = model(x)
    assert (y >= 0).all().item()


def test_default_architecture_layer_count():
    # Default: 7 hidden Linear + 1 output Linear = 8 Linear layers.
    model = DRPRegressor(input_dim=50)
    linears = [m for m in model.net.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) == len(DEFAULT_HIDDEN) + 1
    assert linears[0].in_features == 50
    assert linears[-1].out_features == 1


def test_custom_hidden_dimensions():
    model = DRPRegressor(input_dim=10, hidden=(32, 16))
    x = torch.randn(4, 10)
    y = model(x)
    assert y.shape == (4,)
