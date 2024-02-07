"""Core feature."""

import functools
import operator

from torch import nn

__all__ = ("sequential",)


def sequential(
    data: int = 1,
    layers: int = 3,
    hidden_features: int = 50,
    features: int = 3,
    *,
    dropout: float = 0.0,
    activation: type[nn.Module] | None = None,
) -> nn.Sequential:
    """Linear tanh network.

    Parameters
    ----------
    data : int, optional
        Number of input features, by default 1
    layers : int, optional
        Number of hidden layers, by default 3.
        Must be >= 2.
    hidden_features : int, optional
        Number of hidden units, by default 50.
    features : int, optional
        Number of output features, by default 3.

    dropout : float, optional
        Dropout probability, by default 0.0
    activation : type[`torch.nn.Module`] | None, optional
        Activation function. If `None` (default), uses `torch.nn.Tanh`.

    Returns
    -------
    `torch.nn.Sequential`

    """
    activation_func = nn.Tanh if activation is None else activation

    def make_layer(data: int, hidden_features: int) -> tuple[nn.Module, ...]:
        return (nn.Linear(data, hidden_features), activation_func()) + (
            (nn.Dropout(p=dropout),) if dropout > 0 else ()
        )

    mid_layers = (
        functools.reduce(
            operator.add,
            (make_layer(hidden_features, hidden_features) for _ in range(layers - 2)),
        )
        if layers >= 3  # noqa: PLR2004
        else ()
    )

    return nn.Sequential(
        *make_layer(data, hidden_features),
        *mid_layers,
        nn.Linear(hidden_features, features),
    )
