"""Background models."""


from dataclasses import make_dataclass

from stream_ml.core.background.uniform import Uniform as CoreUniform
from stream_ml.pytorch.background.exponential import Exponential
from stream_ml.pytorch.background.sloped import Sloped
from stream_ml.pytorch.base import ModelBase
from stream_ml.pytorch.typing import Array, NNModel

__all__ = ["Uniform", "Sloped", "Exponential"]


Uniform = make_dataclass(
    "Uniform", [], bases=(CoreUniform[Array, NNModel], ModelBase), unsafe_hash=True
)
