"""Core feature."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Any, ClassVar

from torch import nn

from stream_ml.core.base import ModelBase as CoreModelBase
from stream_ml.core.data import Data
from stream_ml.pytorch.prior.bounds import PriorBounds, SigmoidBounds
from stream_ml.pytorch.typing import Array

__all__: list[str] = []


@dataclass(unsafe_hash=True)
class ModelBase(nn.Module, CoreModelBase[Array]):
    """Model base class."""

    DEFAULT_BOUNDS: ClassVar[PriorBounds] = SigmoidBounds(-inf, inf)

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        nn.Module.__init__(self)  # Needed for PyTorch
        super().__post_init__(*args, **kwargs)

    # ========================================================================
    # ML

    def forward(self, data: Data[Array]) -> Array:
        """Forward pass.

        Parameters
        ----------
        data : Data[Array]
            Input.

        Returns
        -------
        Array
            fraction, mean, sigma
        """
        return self._forward_priors(self.nn(data[self.indep_coord_name]), data)
