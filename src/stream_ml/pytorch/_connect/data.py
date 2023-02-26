"""Core library for stream membership likelihood, with ML."""

from __future__ import annotations

# STDLIB
from typing import Any

import numpy as np

# THIRD-PARTY
import torch as xp

from stream_ml.core.data import ASTYPE_REGISTRY, Data

# --------  Register  ------------------------------------------------------


def _from_ndarray_to_tensor(data: Data[np.ndarray[Any, Any]], /) -> Data[xp.Tensor]:
    """Convert from numpy.ndarray to torch.Tensor."""
    return Data(xp.from_numpy(data.array).float()[..., None], names=data.names)


ASTYPE_REGISTRY[(np.ndarray, xp.Tensor)] = _from_ndarray_to_tensor
