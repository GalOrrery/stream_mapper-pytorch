"""Stream models."""

from dataclasses import field, make_dataclass

import torch as xp

from stream_ml.core.builtin._exponential import Exponential as CoreExponential
from stream_ml.core.builtin._norm import Normal as CoreNormal
from stream_ml.core.builtin._truncnorm import TruncatedNormal as CoreTruncatedNormal
from stream_ml.core.builtin._uniform import Uniform as CoreUniform

from stream_ml.pytorch._base import ModelBase
from stream_ml.pytorch.builtin import _isochrone, compat
from stream_ml.pytorch.builtin._isochrone import *  # noqa: F403
from stream_ml.pytorch.builtin._multinormal import MultivariateNormal
from stream_ml.pytorch.builtin._skewnorm import SkewNormal
from stream_ml.pytorch.builtin._sloped import Sloped
from stream_ml.pytorch.builtin._truncskewnorm import TruncatedSkewNormal
from stream_ml.pytorch.typing import Array, ArrayNamespace, NNModel

__all__ = [
    # modules
    "compat",
    # classes
    "Uniform",
    "Sloped",
    "Exponential",
    "Normal",
    "TruncatedNormal",
    "SkewNormal",
    "TruncatedSkewNormal",
    # -- multivariate
    "MultivariateNormal",
]
__all__ += _isochrone.__all__


Normal = make_dataclass(
    "Normal",
    [("array_namespace", ArrayNamespace[Array], field(default=xp, kw_only=True))],
    bases=(CoreNormal[Array, NNModel], ModelBase),
    unsafe_hash=True,
    repr=False,
)


Uniform = make_dataclass(
    "Uniform",
    [("array_namespace", ArrayNamespace[Array], field(default=xp, kw_only=True))],
    bases=(CoreUniform[Array, NNModel], ModelBase),
    unsafe_hash=True,
    repr=False,
)


Exponential = make_dataclass(
    "Exponential",
    [("array_namespace", ArrayNamespace[Array], field(default=xp, kw_only=True))],
    bases=(CoreExponential[Array, NNModel], ModelBase),
    unsafe_hash=True,
    repr=False,
)


TruncatedNormal = make_dataclass(
    "TruncatedNormal",
    [("array_namespace", ArrayNamespace[Array], field(default=xp, kw_only=True))],
    bases=(CoreTruncatedNormal[Array, NNModel], ModelBase),
    unsafe_hash=True,
    repr=False,
)


# This version is not numerically stable in the far tails
# SkewNormal = make_dataclass(
#     "SkewNormal",
#     [("array_namespace", ArrayNamespace[Array], field(default=xp, kw_only=True))],
#     bases=(CoreSkewNormal[Array, NNModel], ModelBase),
#     unsafe_hash=True,
# )


# TruncatedSkewNormal = make_dataclass(
#     "TruncatedSkewNormal",
#     [("array_namespace", ArrayNamespace[Array], field(default=xp, kw_only=True))],
#     bases=(CoreTruncatedSkewNormal[Array, NNModel], ModelBase),
#     unsafe_hash=True,
# )
