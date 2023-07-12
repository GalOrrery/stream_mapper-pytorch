"""Core library for stream membership likelihood, with ML."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as xp

from stream_ml.core._data import ASTYPE_REGISTRY, Data

# --------  Register  ------------------------------------------------------


def _from_ndarray_to_tensor(data: Data[np.ndarray[Any, Any]], /) -> Data[xp.Tensor]:
    """Convert from numpy.ndarray to torch.Tensor."""
    return Data(xp.from_numpy(data.array).float(), names=data.names)


ASTYPE_REGISTRY[(np.ndarray, xp.Tensor)] = _from_ndarray_to_tensor


try:
    import asdf
except ImportError:
    pass
else:

    def _from_ndarraytype_to_tensor(
        data: Data[np.ndarray[Any, Any]], /
    ) -> Data[xp.Tensor]:
        """Convert from numpy.ndarray to torch.Tensor."""
        return Data(xp.from_numpy(np.asarray(data.array)).float(), names=data.names)

    ASTYPE_REGISTRY[
        (asdf.tags.core.ndarray.NDArrayType, xp.Tensor)
    ] = _from_ndarraytype_to_tensor
