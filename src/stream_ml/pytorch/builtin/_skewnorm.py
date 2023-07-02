"""Gaussian stream model."""

from __future__ import annotations

__all__: list[str] = []

from dataclasses import dataclass
from typing import TYPE_CHECKING

from stream_ml.pytorch._base import ModelBase
from stream_ml.pytorch.builtin._stats.skewnorm import logpdf

if TYPE_CHECKING:
    from stream_ml.core.params import Params

    from stream_ml.pytorch import Data
    from stream_ml.pytorch.typing import Array


@dataclass(unsafe_hash=True)
class SkewNormal(ModelBase):
    r"""1D Gaussian with mixture weight.

    :math:`(weight, \mu, \ln\sigma)(\phi1)`
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        # Validate the coord_names
        if len(self.coord_names) != 1:
            msg = "Only one coordinate is supported, e.g ('phi2',)"
            raise ValueError(msg)
        if self.coord_err_names is not None and len(self.coord_err_names) != 1:
            msg = "Only one coordinate error is supported, e.g ('phi2_err',)"
            raise ValueError(msg)

    def ln_likelihood(
        self, mpars: Params[Array], /, data: Data[Array], **kwargs: Array
    ) -> Array:
        """Log-likelihood of the distribution.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array]
            Data (phi1, phi2).
        **kwargs : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        c = self.coord_names[0]
        return logpdf(
            data[c],
            loc=mpars[c, "mu"],
            sigma=self.xp.exp(mpars[c, "ln-sigma"]),
            skew=mpars[c, "skew"],
        )
