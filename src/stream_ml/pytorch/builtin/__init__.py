"""Stream models."""

from dataclasses import make_dataclass

from stream_ml.core.builtin.uniform import Uniform as CoreUniform
from stream_ml.pytorch.base import ModelBase
from stream_ml.pytorch.builtin.exponential import Exponential
from stream_ml.pytorch.builtin.multinormal import (
    MultivariateMissingNormal,
    MultivariateNormal,
)
from stream_ml.pytorch.builtin.normal import Normal
from stream_ml.pytorch.builtin.sloped import Sloped
from stream_ml.pytorch.typing import Array, NNModel

__all__ = [
    "Uniform",
    "Sloped",
    "Exponential",
    "Normal",
    "MultivariateNormal",
    "MultivariateMissingNormal",
]


Uniform = make_dataclass(
    "Uniform", [], bases=(CoreUniform[Array, NNModel], ModelBase), unsafe_hash=True
)
