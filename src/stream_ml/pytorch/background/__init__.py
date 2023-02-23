"""Background models."""


from dataclasses import dataclass

from stream_ml.core.background.uniform import Uniform as CoreUniform
from stream_ml.pytorch.base import ModelBase
from stream_ml.pytorch.typing import Array, NNModel

from .exponential import Exponential
from .sloped import Sloped

__all__ = ["Uniform", "Sloped", "Exponential"]


@dataclass(unsafe_hash=True)
class Uniform(CoreUniform[Array, NNModel], ModelBase):
    """Uniform background model."""
