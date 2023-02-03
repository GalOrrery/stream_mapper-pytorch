"""Core feature."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch as xp

from stream_ml.core.data import Data
from stream_ml.core.prior.bounds import PriorBounds as CorePriorBounds
from stream_ml.core.utils.funcs import within_bounds
from stream_ml.pytorch.typing import Array
from stream_ml.pytorch.utils.sigmoid import scaled_sigmoid

if TYPE_CHECKING:
    from stream_ml.core.api import Model
    from stream_ml.core.params.core import Params

__all__: list[str] = []


@dataclass(frozen=True)
class PriorBounds(CorePriorBounds[Array]):
    """Base class for prior bounds."""

    def __post_init__(self) -> None:
        """Post-init."""
        object.__setattr__(self, "_lower_torch", xp.asarray([self.lower]))
        object.__setattr__(self, "_upper_torch", xp.asarray([self.upper]))

    def logpdf(
        self,
        mpars: Params[Array],
        data: Data[Array],
        model: Model[Array],
        current_lnpdf: Array | None = None,
        /,
    ) -> Array | float:
        """Evaluate the logpdf."""
        if self.param_name is None:
            msg = "need to set param_name"
            raise ValueError(msg)

        bp = xp.zeros_like(mpars[self.param_name])
        bp[~within_bounds(mpars[self.param_name], self.lower, self.upper)] = -xp.inf
        return bp

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
