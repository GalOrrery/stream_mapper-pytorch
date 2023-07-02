"""Gaussian stream model."""

from __future__ import annotations

__all__: list[str] = []

from dataclasses import field, make_dataclass

import torch as xp

from stream_ml.core.builtin._norm import Normal as CoreNormal

from stream_ml.pytorch._base import ModelBase
from stream_ml.pytorch.typing import Array, ArrayNamespace, NNModel

Normal = make_dataclass(
    "Normal",
    [("array_namespace", ArrayNamespace[Array], field(default=xp, kw_only=True))],
    bases=(CoreNormal[Array, NNModel], ModelBase),
    unsafe_hash=True,
)
