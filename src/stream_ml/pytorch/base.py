"""Core feature."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from math import inf
from typing import Any, ClassVar

import torch as xp
from torch import nn

from stream_ml.core.base import ModelBase as CoreModelBase
from stream_ml.core.data import Data
from stream_ml.core.params import Params
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

        TODO: this is returning NaN for some reason

        .. code-block:: python

            where = reduce(
                xp.logical_or,
                (~within_bounds(data[k], *v) for k, v in self.coord_bounds.items()),
            )
            lnp[where] = -xp.inf

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
        # TODO! move this to be a method on coord_bounds
        lnp = xp.zeros((len(data), 1))
        return lnp  # noqa: RET504

    # ========================================================================
    # ML

    def _forward_prior(self, out: Array, data: Data[Array]) -> Array:
        """Forward pass.

        Parameters
        ----------
        out : Array
            Input.
        data : Data[Array]
            Data.

        Returns
        -------
        Array
            Same as input.
        """
        for bnd in self.param_bounds.flatvalues():
            out = bnd(out, data, self)
        return out

    @abstractmethod
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
        raise NotImplementedError
