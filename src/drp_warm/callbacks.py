"""Lightweight callbacks for the training loops.

The continue phase needs to stop as soon as a target validation metric is
reached (rather than running to a fixed epoch budget). This module provides
a small, framework-free stop condition that the trainer checks after each
epoch's evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

StopMode = Literal["below", "above"]


@dataclass(frozen=True)
class EarlyStoppingByMetric:
    """Stop when a monitored metric crosses a threshold.

    For loss-like metrics (lower is better), use ``mode="below"`` and pass
    the target as ``ref_min × (1 + margin)``. For score-like metrics
    (higher is better), use ``mode="above"``.
    """

    target: float
    mode: StopMode = "below"

    def should_stop(self, value: float) -> bool:
        if self.mode == "below":
            return value <= self.target
        return value >= self.target
