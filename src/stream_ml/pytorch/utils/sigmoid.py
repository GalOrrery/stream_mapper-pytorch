"""Core feature."""

from __future__ import annotations

# STDLIB
from typing import TYPE_CHECKING

# THIRD-PARTY
import torch as xp

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.pytorch.typing import Array

__all__: list[str] = []

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

    See Also
    --------
    stream_ml.core.utils.map_to_range
        Maps ``[min(x), max(x)]`` to range ``[lower, upper]``.
    """
    if xp.isneginf(lower) and xp.isposinf(upper):
        return x
    elif xp.isinf(lower) or xp.isinf(upper):
        raise NotImplementedError

    return xp.sigmoid(x) * (upper - lower) + lower
