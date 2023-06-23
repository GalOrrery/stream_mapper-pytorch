"""Built-in background models."""

__all__: list[str] = []

from dataclasses import field, make_dataclass

import torch as xp

from stream_ml.core.builtin._exponential import Exponential as CoreExponential
from stream_ml.core.typing import ArrayNamespace

from stream_ml.pytorch._base import ModelBase
from stream_ml.pytorch.typing import Array, NNModel

Exponential = make_dataclass(
    "Exponential",
    [("array_namespace", ArrayNamespace[Array], field(default=xp, kw_only=True))],
    bases=(CoreExponential[Array, NNModel], ModelBase),
    unsafe_hash=True,
)
