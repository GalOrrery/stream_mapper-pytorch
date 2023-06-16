"""Stream models."""

__all__ = [
    # modules
    "compat",
    # classes
    "Uniform",
    "Sloped",
    "Exponential",
    "IsochroneMVNorm",
    "Normal",
    "TruncatedNormal",
    "SkewNormal",
    "TruncatedSkewNormal",
    "MultivariateNormal",
    "MultivariateMissingNormal",
]

from dataclasses import field, make_dataclass

import torch as xp

from stream_ml.core.builtin._uniform import Uniform as CoreUniform
from stream_ml.core.typing import ArrayNamespace

from stream_ml.pytorch._base import ModelBase
from stream_ml.pytorch.builtin import compat
from stream_ml.pytorch.builtin._exponential import Exponential
from stream_ml.pytorch.builtin._isochrone import IsochroneMVNorm
from stream_ml.pytorch.builtin._multinormal import (
    MultivariateMissingNormal,
    MultivariateNormal,
)
from stream_ml.pytorch.builtin._norm import Normal
from stream_ml.pytorch.builtin._skewnorm import SkewNormal
from stream_ml.pytorch.builtin._sloped import Sloped
from stream_ml.pytorch.builtin._truncnorm import TruncatedNormal
from stream_ml.pytorch.builtin._truncskewnorm import TruncatedSkewNormal
from stream_ml.pytorch.typing import Array, NNModel

Uniform = make_dataclass(
    "Uniform",
    [("array_namespace", ArrayNamespace[Array], field(default=xp, kw_only=True))],
    bases=(CoreUniform[Array, NNModel], ModelBase),
    unsafe_hash=True,
)
