"""Core feature."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch as xp

from stream_ml.core.data import Data
from stream_ml.core.prior.bounds import PriorBounds as CorePriorBounds
from stream_ml.pytorch.typing import Array
from stream_ml.pytorch.utils.sigmoid import scaled_sigmoid

if TYPE_CHECKING:
    from stream_ml.core.api import Model

__all__: list[str] = []


@dataclass(frozen=True)
class PriorBounds(CorePriorBounds[Array]):
    """Base class for prior bounds."""

    def __post_init__(self) -> None:
        """Post-init."""
        object.__setattr__(self, "_lower_torch", xp.asarray([self.lower]))
        object.__setattr__(self, "_upper_torch", xp.asarray([self.upper]))

    @abstractmethod
    def __call__(self, pred: Array, data: Data[Array], model: Model[Array], /) -> Array:
        """Evaluate the forward step in the prior."""
        ...


@dataclass(frozen=True)
class SigmoidBounds(PriorBounds):
    """Base class for prior bounds."""

    def __call__(self, pred: Array, data: Data[Array], model: Model[Array], /) -> Array:
        """Evaluate the forward step in the prior."""
        pred = pred.clone()

        # Get column
        col = model.param_names.flats.index(self.param_name)
        pred[:, col] = scaled_sigmoid(
            pred[:, col], lower=self._lower_torch, upper=self._upper_torch
        )
        return pred
