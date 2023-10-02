"""Core library for stream membership likelihood, with ML."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import torch as xp

from stream_ml.core._data import ASTYPE_REGISTRY, Data

# --------  Register  ------------------------------------------------------


def _from_tensor_to_tensor(data: Data[xp.Tensor], /, **kwargs: Any) -> Data[xp.Tensor]:
    """Convert from numpy.ndarray to torch.Tensor."""
    return replace(data, array=data.array.to(**kwargs))


ASTYPE_REGISTRY[(xp.Tensor, xp.Tensor)] = _from_tensor_to_tensor


def _from_ndarray_to_tensor(
    data: Data[np.ndarray[Any, Any]], /, **kwargs: Any
) -> Data[xp.Tensor]:
    """Convert from numpy.ndarray to torch.Tensor."""
    return replace(data, array=xp.asarray(data.array, **kwargs))


ASTYPE_REGISTRY[(np.ndarray, xp.Tensor)] = _from_ndarray_to_tensor


def _from_tensor_to_ndarray(
    data: Data[xp.Tensor], /, **kwargs: Any
) -> Data[np.ndarray[Any, Any]]:
    """Convert from torch.Tensor to numpy.ndarray."""
    return replace(data, array=np.asarray(data.array, **kwargs))


ASTYPE_REGISTRY[(xp.Tensor, np.ndarray)] = _from_tensor_to_ndarray  # type: ignore[assignment]


try:
    import asdf
except ImportError:
    pass
else:

    def _from_ndarraytype_to_tensor(
        data: Data[np.ndarray[Any, Any]], /, **kwargs: Any
    ) -> Data[xp.Tensor]:
        """Convert from numpy.ndarray to torch.Tensor."""
        array = np.array(data.array, copy=True, subok=False)
        array.flags.writeable = True
        return replace(data, array=xp.asarray(array, **kwargs))

    ASTYPE_REGISTRY[
        (asdf.tags.core.ndarray.NDArrayType, xp.Tensor)
    ] = _from_ndarraytype_to_tensor
