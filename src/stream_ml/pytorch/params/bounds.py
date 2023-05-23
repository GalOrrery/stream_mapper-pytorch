"""Core feature."""

from __future__ import annotations

__all__ = [
    # core
    "SigmoidBounds",
    # pytorch
    "ParameterBounds",
    "NoBounds",
    "ClippedBounds",
]

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch as xp

from stream_ml.core.params.bounds import ClippedBounds, NoBounds, ParameterBounds

from stream_ml.pytorch.typing import Array, NNModel

if TYPE_CHECKING:
    from stream_ml.core import ModelAPI
    from stream_ml.core.data import Data

_0 = xp.asarray(0)
_1 = xp.asarray(1)


def scaled_sigmoid(x: Array, /, lower: Array = _0, upper: Array = _1) -> Array:
    """Sigmoid function mapping ``(-inf, inf)`` to ``(lower, upper)``.

    Output for (lower, upper) is defined as:
    - If (finite, finite), then this is a scaled sigmoid function.
    - If (-inf, inf) then this is the identity function.
    - Not implemented for (+/- inf, any), (any, +/- inf)

    Parameters
    ----------
    x : Array
        X.
    lower : Array
        Lower.
    upper : Array
        Upper.

    Returns
    -------
    Array
    """
    if xp.isneginf(lower) and xp.isposinf(upper):
        return x
    elif xp.isinf(lower) or xp.isinf(upper):
        raise NotImplementedError

    return xp.sigmoid(x) * (upper - lower) + lower


@dataclass(frozen=True)
class SigmoidBounds(ParameterBounds[Array]):
    """Base class for prior bounds."""

    def __post_init__(self) -> None:
        """Post-init."""
        super().__post_init__()

        object.__setattr__(self, "lower", xp.asarray([self.lower]))
        object.__setattr__(self, "upper", xp.asarray([self.upper]))

    def __call__(
        self, pred: Array, data: Data[Array], model: ModelAPI[Array, NNModel], /
    ) -> Array:
        """Evaluate the forward step in the prior."""
        pred = pred.clone()

        # Get column
        col = model.params.flatskeys().index(self.param_name)
        pred[:, col] = scaled_sigmoid(pred[:, col], *self.scaled_bounds)
        return pred
