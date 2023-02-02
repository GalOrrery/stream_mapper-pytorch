"""Core feature."""

from __future__ import annotations

# THIRD-PARTY
import torch as xp

# LOCAL
from stream_ml.pytorch.typing import Array

__all__: list[str] = []


def within_bounds(
    value: Array,
    /,
    lower_bound: Array | float | None,
    upper_bound: Array | float | None,
    *,
    lower_inclusive: bool = True,
    upper_inclusive: bool = True,
) -> Array:
    """Check if a value is within the given bounds.

    Parameters
    ----------
    value : ndarray
        Value to check.
    lower_bound, upper_bound : float | None
        Bounds to check against.
    lower_inclusive, upper_inclusive : bool, optional
        Whether to include the bounds in the check, by default `True`.

    Returns
    -------
    ndarray
        Boolean array indicating whether the value is within the bounds.
    """
    inbounds = xp.ones_like(value, dtype=xp.bool)
    if lower_bound is not None:
        inbounds &= value >= lower_bound if lower_inclusive else value > lower_bound
    if upper_bound is not None:
        inbounds &= value <= upper_bound if upper_inclusive else value < upper_bound

    return inbounds
