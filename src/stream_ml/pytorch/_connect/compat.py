"""Functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch as xp

from stream_ml.core.utils.compat import array_at

if TYPE_CHECKING:
    from stream_ml.pytorch.typing import Array


class ArrayAt:
    """Array at index.

    This is to emulate the `jax.numpy.ndarray.at` method.
    """

    def __init__(self, array: Array, idx: Any) -> None:
        """Initialize."""
        self.array = array
        self.idx = idx

    def set(self, value: Array | float) -> Array:  # noqa: A003
        """Set the value at the index, in-place."""
        self.array[self.idx] = value
        return self.array


@array_at.register(xp.Tensor)  # type: ignore[misc]
def _array_at_pytorch(array: Array, idx: Any, /, *, inplace: bool = True) -> ArrayAt:
    """Get the array at the index.

    This is to emulate the `jax.numpy.ndarray.at` method.

    Parameters
    ----------
    array : Array
        Array to get the value at the index.
    idx : Any
        Index to get the value at.

    inplace : bool, optional
        Whether to set the value in-place, by default `False`.

    Returns
    -------
    ArrayAt[Array]
        Setter.
    """
    return ArrayAt(array if inplace else array.clone(), idx)
