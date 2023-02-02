"""Core feature."""

import functools
import operator

from torch import nn

__all__ = ["lin_tanh"]


def lin_tanh(n_in: int = 1, n_hidden: int = 50, n_layers: int = 3, n_out: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(n_in, n_hidden),
        nn.Tanh(),
        *functools.reduce(
            operator.add,
            (
                (nn.Linear(n_hidden, n_hidden), nn.Tanh())
                for _ in range(n_layers - 2)
            ),
        ),
        nn.Linear(n_hidden, n_out),
    )