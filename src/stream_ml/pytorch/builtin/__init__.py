"""Stream models."""

from dataclasses import field, make_dataclass

import torch as xp

from stream_ml.core.builtin._uniform import Uniform as CoreUniform
from stream_ml.core.typing import ArrayNamespace
from stream_ml.pytorch._base import ModelBase
from stream_ml.pytorch.builtin._exponential import Exponential
from stream_ml.pytorch.builtin._isochrone import IsochroneMVNorm
from stream_ml.pytorch.builtin._multinormal import (
    MultivariateMissingNormal,
    MultivariateNormal,
)
from stream_ml.pytorch.builtin._normal import Normal
from stream_ml.pytorch.builtin._sloped import Sloped
from stream_ml.pytorch.builtin._wrapper import WithWeightModel
from stream_ml.pytorch.typing import Array, NNModel

__all__ = [
    "Uniform",
    "Sloped",
    "Exponential",
    "IsochroneMVNorm",
    "Normal",
    "MultivariateNormal",
    "MultivariateMissingNormal",
    "WithWeightModel",
]


Uniform = make_dataclass(
    "Uniform",
    [("array_namespace", ArrayNamespace[Array], field(default=xp, kw_only=True))],
    bases=(CoreUniform[Array, NNModel], ModelBase),
    unsafe_hash=True,
)
