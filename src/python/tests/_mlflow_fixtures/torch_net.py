"""Tiny ``nn.Module`` defined outside ``__main__`` for the pytorch e2e tests.

The pytorch flavor pickles by reference: the class must be importable from a
module other than ``__main__`` to exercise the ``code_paths`` shipping path.
"""

from __future__ import annotations

import torch  # type: ignore[import-not-found]
from torch import nn  # type: ignore[import-not-found]


N_FEATURES = 4


class TinyNet(nn.Module):
    """One linear layer; deterministic by construction with fixed weights."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(N_FEATURES, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
