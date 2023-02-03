"""Core feature."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from math import inf
from typing import Any, ClassVar

import torch as xp
from torch import nn

from stream_ml.core.api import WEIGHT_NAME
from stream_ml.core.base import ModelBase as CoreModelBase
from stream_ml.core.data import Data
from stream_ml.core.params import Params
from stream_ml.core.utils.compat import array_at
from stream_ml.core.utils.funcs import within_bounds
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
    # Statistics

    def _ln_prior_coord_bnds(self, mpars: Params[Array], data: Data[Array]) -> Array:
        """Elementwise log prior for coordinate bounds.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array]
            Data.

        Returns
        -------
        Array
            Zero everywhere except where the data are outside the
            coordinate bounds, where it is -inf.
        """
        lnp = self.xp.zeros_like(mpars[(WEIGHT_NAME,)])
        where = reduce(
            xp.logical_or,
            (~within_bounds(data[k], *v) for k, v in self.coord_bounds.items()),
        )
        return array_at(lnp, where).set(-self.xp.inf)

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
