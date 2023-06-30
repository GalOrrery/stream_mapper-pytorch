"""Core feature."""

import functools
import operator

from torch import nn

__all__ = ["lin_tanh"]


def lin_tanh(
    n_in: int = 1,
    n_hidden: int = 50,
    n_layers: int = 3,
    n_out: int = 3,
    dropout: float = 0.0,
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

    dropout : float, optional
        Dropout probability, by default 0.0

    Returns
    -------
    `torch.nn.Sequential`
    """

    def make_layer(n_in: int, n_hidden: int) -> tuple[nn.Module, ...]:
        return (nn.Linear(n_in, n_hidden), nn.Tanh()) + (
            (nn.Dropout(p=dropout),) if dropout > 0 else ()
        )

    mid_layers = (
        functools.reduce(
            operator.add,
            (make_layer(n_hidden, n_hidden) for _ in range(n_layers - 2)),
        )
        if n_layers >= 3  # noqa: PLR2004
        else ()
    )

    return nn.Sequential(
        *make_layer(n_in, n_hidden),
        *mid_layers,
        nn.Linear(n_hidden, n_out),
    )
