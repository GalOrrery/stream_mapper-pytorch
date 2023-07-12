"""Core library for stream membership likelihood, with ML."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch as xp

from stream_ml.core.utils.scale._api import ASTYPE_REGISTRY
from stream_ml.core.utils.scale._standard import StandardScaler

__all__: list[str] = []


#####################################################################
# StandardScaler


def standard_scaler_astype_tensor(
    scaler: StandardScaler[Any],
) -> StandardScaler[xp.Tensor]:
    """Register the `StandardScaler` class for `numpy.ndarray`."""
    return replace(
        scaler,
        mean=xp.asarray(scaler.mean),
        scale=xp.asarray(scaler.scale),
        names=scaler.names,
    )


ASTYPE_REGISTRY[(StandardScaler, xp.Tensor)] = standard_scaler_astype_tensor  # type: ignore[assignment]  # noqa: E501
