"""Background models."""


from dataclasses import make_dataclass

from stream_ml.core.background.uniform import Uniform as CoreUniform
from stream_ml.pytorch.base import ModelBase
from stream_ml.pytorch.typing import Array, NNModel

from .exponential import Exponential
from .sloped import Sloped

__all__ = ["Uniform", "Sloped", "Exponential"]


Uniform = make_dataclass(
    "Uniform", [], bases=(CoreUniform[Array, NNModel], ModelBase), unsafe_hash=True
)
