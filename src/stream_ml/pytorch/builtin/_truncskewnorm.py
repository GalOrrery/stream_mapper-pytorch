"""Gaussian stream model."""

from __future__ import annotations

__all__: list[str] = []

from dataclasses import dataclass
from typing import TYPE_CHECKING

from stream_ml.pytorch.builtin._skewnorm import SkewNormal
from stream_ml.pytorch.builtin._stats.truncskewnorm import logpdf
from stream_ml.pytorch.builtin._truncnorm import TruncatedNormal

if TYPE_CHECKING:
    from stream_ml.core.data import Data
    from stream_ml.core.params import Params

    from stream_ml.pytorch.typing import Array


@dataclass(unsafe_hash=True)
class TruncatedSkewNormal(TruncatedNormal, SkewNormal):
    r"""1D Gaussian with mixture weight.

    :math:`(weight, \mu, \ln\sigma)(\phi1)`
    """

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
            a=self.coord_bounds[self.coord_names[0]][0],
            b=self.coord_bounds[self.coord_names[0]][1],
            nil=-100.0,
        )
