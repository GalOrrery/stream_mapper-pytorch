"""Functions."""

from typing import Any

import torch as xp

from stream_ml.core.utils.compat import ArrayAt as CoreArrayAt
from stream_ml.core.utils.compat import array_at
from stream_ml.pytorch.typing import Array


class ArrayAt(CoreArrayAt[Array]):
    """Array at index.

    This is to emulate the `jax.numpy.ndarray.at` method.
    """

    def __init__(self, array: Array, idx: Any) -> None:
        """Initialize."""
        self.array = array
        self.idx = idx

    def set(self, value: Array) -> Array:  # noqa: A003
        """Set the value at the index, in-place."""
        self.array[self.idx] = value
        return self.array


@array_at.register(xp.Tensor)
def _array_at_pytorch(array: Array, idx: Any) -> ArrayAt:
    """Get the array at the index.

    This is to emulate the `jax.numpy.ndarray.at` method.

    Parameters
    ----------
    array : Array
        Array to get the value at the index.
    idx : Any
        Index to get the value at.

    Returns
    -------
    ArrayAt[Array]
        Setter.
    """
    return ArrayAt(array, idx)
