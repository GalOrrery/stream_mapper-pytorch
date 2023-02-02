"""Core library for stream membership likelihood, with ML."""

from __future__ import annotations

# STDLIB
from typing import Any

import numpy as np

# THIRD-PARTY
import torch as xp

from stream_ml.core.data import ARRAY_HOOK, DATA_HOOK, TO_FORMAT_REGISTRY, Data

# LOCAL
from stream_ml.pytorch.typing import Array

# --------  Register  ------------------------------------------------------


def _data_hook(data: Data[Array], /) -> Data[Array]:
    if isinstance(data, Data) and data.array.ndim == 1:
        object.__setattr__(data, "array", data.array[:, None])
    return data


DATA_HOOK[Array] = _data_hook


def _array_hook(array: Array, /) -> Array:
    if isinstance(array, Array) and array.ndim == 1:
        return array[:, None]
    return array


ARRAY_HOOK[Array] = _array_hook


def _from_ndarray_to_tensor(data: Data[np.ndarray[Any, Any]], /) -> Data[xp.Tensor]:
    """Convert from numpy.ndarray to torch.Tensor."""
    return Data(xp.from_numpy(data.array).float(), names=data.names)


TO_FORMAT_REGISTRY[(np.ndarray, xp.Tensor)] = _from_ndarray_to_tensor
