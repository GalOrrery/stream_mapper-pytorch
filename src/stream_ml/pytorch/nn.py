"""Core feature."""

import functools
import operator

from torch import nn

__all__ = ["lin_tanh"]


def lin_tanh(
    n_in: int = 1, n_hidden: int = 50, n_layers: int = 3, n_out: int = 3
) -> nn.Sequential:
    """Linear tanh network.

    Parameters
    ----------
    n_in : int, optional
        Number of input features, by default 1
    n_hidden : int, optional
        Number of hidden units, by default 50.
    n_layers : int, optional
        Number of hidden layers, by default 3.
        Must be >= 2.
    n_out : int, optional
        Number of output features, by default 3.
    """
    midlayers = (
        functools.reduce(
            operator.add,
            ((nn.Linear(n_hidden, n_hidden), nn.Tanh()) for _ in range(n_layers - 2)),
        )
        if n_layers >= 3  # noqa: PLR2004
        else ()
    )

    return nn.Sequential(
        nn.Linear(n_in, n_hidden),
        nn.Tanh(),
        *midlayers,
        nn.Linear(n_hidden, n_out),
    )
