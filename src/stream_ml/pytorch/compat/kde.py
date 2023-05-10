"""Core feature."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

import torch as xp

from stream_ml.core.params.names import ParamNamesField
from stream_ml.pytorch._base import ModelBase

__all__: list[str] = []


if TYPE_CHECKING:
    from scipy.stats import gaussian_kde

    from stream_ml.core.data import Data
    from stream_ml.core.params import Params
    from stream_ml.pytorch.typing import Array


@dataclass(unsafe_hash=True)
class KDEModel(ModelBase):
    """Normalizing flow model."""

    _: KW_ONLY
    kernel: gaussian_kde
    param_names: ParamNamesField = ParamNamesField(())

    def __post_init__(self) -> None:
        """Post-initialization hook."""
        if self.net is not None:
            msg = "Cannot pass `net` to KDEModel."
            raise ValueError(msg)

    def ln_likelihood(
        self, mpars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Log-likelihood of the array.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters. The flow has an internal weight, so we don't use the
            weight, if passed.
        data : Data[Array]
            Data (phi1, phi2).

        **kwargs : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        with xp.no_grad():
            return xp.log(xp.asarray(self.kernel(data[:, self.coord_names, 0].T)))[
                :, None
            ]

    def forward(self, data: Data[Array]) -> Array:
        """Forward pass.

        Parameters
        ----------
        data : Data[Array]
            Input. Only uses the first argument.

        Returns
        -------
        Array
        """
        return xp.asarray([])
