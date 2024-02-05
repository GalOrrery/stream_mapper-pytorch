"""Core feature."""

from __future__ import annotations

__all__ = (
    # core
    "ParameterBounds",
    "NoBounds",
    "ClippedBounds",
    # pytorch
    "SigmoidBounds",
)

from dataclasses import KW_ONLY, dataclass, field, make_dataclass
from typing import TYPE_CHECKING

import torch as xp

from stream_mapper.core.params.bounds import ClippedBounds as CoreClippedBounds
from stream_mapper.core.params.bounds import NoBounds as CoreNoBounds
from stream_mapper.core.params.bounds import ParameterBounds

from stream_mapper.pytorch.typing import Array, ArrayNamespace, NNModel

if TYPE_CHECKING:
    from stream_mapper.core import Data, ModelAPI
    from stream_mapper.core.params import ParamScaler


NoBounds = make_dataclass(
    "NoBounds",
    [("array_namespace", ArrayNamespace[Array], field(default="torch", kw_only=True))],
    bases=(CoreNoBounds[Array],),
    frozen=True,
    repr=False,
)


ClippedBounds = make_dataclass(
    "ClippedBounds",
    [("array_namespace", ArrayNamespace[Array], field(default="torch", kw_only=True))],
    bases=(CoreClippedBounds[Array],),
    frozen=True,
    repr=False,
)


# ==============================================================================


def scaled_sigmoid(x: Array, /, lower: Array, upper: Array) -> Array:
    """Sigmoid function mapping ``(-inf, inf)`` to ``(lower, upper)``.

    Output for (lower, upper) is defined as:
    - If (finite, finite), then this is a scaled sigmoid function.
    - If (-inf, inf) then this is the identity function.
    - Not implemented for (+/- inf, any), (any, +/- inf)

    Parameters
    ----------
    x : Array
        X.
    lower, upper : Array
        Bounds.

    Returns
    -------
    Array

    """
    if xp.isneginf(lower) and xp.isposinf(upper):
        return x
    elif xp.isinf(lower) or xp.isinf(upper):
        raise NotImplementedError

    return xp.sigmoid(x) * (upper - lower) + lower


@dataclass(frozen=True, repr=False)
class SigmoidBounds(ParameterBounds[Array]):
    """Base class for prior bounds."""

    _: KW_ONLY
    array_namespace: ArrayNamespace[Array] = xp

    def __post_init__(self, scaler: ParamScaler[Array] | None) -> None:
        """Post-init."""
        super().__post_init__(scaler)

        # Convert to array
        object.__setattr__(self, "lower", xp.asarray([self.lower]))
        object.__setattr__(self, "upper", xp.asarray([self.upper]))

    def __call__(
        self, pred: Array, data: Data[Array], model: ModelAPI[Array, NNModel], /
    ) -> Array:
        """Evaluate the forward step in the prior."""
        pred = pred.clone()
        col = model.params.flatskeys().index(self.param_name)
        pred[:, col] = scaled_sigmoid(pred[:, col], *self.scaled_bounds)
        return pred
