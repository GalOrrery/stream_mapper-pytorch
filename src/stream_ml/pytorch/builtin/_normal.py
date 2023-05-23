"""Gaussian stream model."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from stream_ml.pytorch._base import ModelBase

if TYPE_CHECKING:
    from stream_ml.core.data import Data
    from stream_ml.core.params import Params
    from stream_ml.core.typing import ArrayNamespace

    from stream_ml.pytorch.typing import Array

__all__: list[str] = []


_logsqrt2pi = math.log(2 * math.pi) / 2
_sqrt2 = math.sqrt(2)


def norm_logpdf(
    value: Array, loc: Array, sigma: Array, *, xp: ArrayNamespace[Array]
) -> Array:
    r"""Log of the probability density function of the normal distribution.

    Parameters
    ----------
    value : Array
        Value at which to evaluate the PDF.
    loc : Array
        Mean of the distribution.
    sigma : Array
        variance of the distribution.

    xp : ArrayNamespace, keyword-only
        Array namespace.

    Returns
    -------
    Array
        Log of the PDF.
    """
    return -0.5 * ((value - loc) / sigma) ** 2 - xp.log(xp.abs(sigma)) - _logsqrt2pi


@dataclass(unsafe_hash=True)
class Normal(ModelBase):
    r"""1D Gaussian with mixture weight.

    :math:`(weight, \mu, \sigma)(\phi1)`

    Parameters
    ----------
    n_layers : int, optional
        Number of hidden layers, by default 3.
    hidden_features : int, optional
        Number of hidden features, by default 50.
    sigma_upper_limit : float, optional keyword-only
        Upper limit on sigma, by default 0.3.
    fraction_upper_limit : float, optional keyword-only
        Upper limit on fraction, by default 0.45.s
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        # Validate the coord_names
        if len(self.coord_names) != 1:
            msg = "Only one coordinate is supported, e.g ('phi2',)"
            raise ValueError(msg)

    # ========================================================================
    # Statistics

    def ln_likelihood(
        self, mpars: Params[Array], data: Data[Array], **kwargs: Array
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
        return norm_logpdf(
            data[c],
            mpars[c, "mu"],
            self.xp.clip(mpars[c, "sigma"], 1e-10),
            xp=self.xp,
        )


# ============================================================================


def log_truncation_term(
    ab: tuple[Array, Array], /, loc: Array, sigma: Array, *, xp: ArrayNamespace[Array]
) -> Array:
    """Log of integral from a to b of normal."""
    erfa = xp.erf((ab[0] - loc) / sigma / _sqrt2)  # type: ignore[attr-defined]
    erfb = xp.erf((ab[1] - loc) / sigma / _sqrt2)  # type: ignore[attr-defined]
    return xp.log(erfb - erfa) - xp.log(4)


def truncnorm_logpdf(
    value: Array,
    /,
    loc: Array,
    sigma: Array,
    ab: tuple[float | Array, float | Array],
    *,
    xp: ArrayNamespace[Array],
) -> Array:
    return norm_logpdf(value, loc=loc, sigma=sigma, xp=xp) - log_truncation_term(
        ab, loc=loc, sigma=sigma, xp=xp
    )


@dataclass(unsafe_hash=True)
class TruncatedNormal(Normal):
    r"""1D Gaussian with mixture weight.

    :math:`(weight, \mu, \sigma)(\phi1)`

    Parameters
    ----------
    n_layers : int, optional
        Number of hidden layers, by default 3.
    hidden_features : int, optional
        Number of hidden features, by default 50.
    sigma_upper_limit : float, optional keyword-only
        Upper limit on sigma, by default 0.3.
    fraction_upper_limit : float, optional keyword-only
        Upper limit on fraction, by default 0.45.s
    """

    # ========================================================================
    # Statistics

    def ln_likelihood(
        self, mpars: Params[Array], data: Data[Array], **kwargs: Array
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
        return truncnorm_logpdf(
            data[c],
            loc=mpars[c, "mu"],
            sigma=self.xp.clip(mpars[c, "sigma"], 1e-10),
            ab=self.coord_bounds[self.coord_names[0]],
            xp=self.xp,
        )
