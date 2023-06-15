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


log2pi = math.log(2 * math.pi)
sqrt2 = math.sqrt(2)


def norm_logpdf(
    x: Array, loc: Array, sigma: Array, *, xp: ArrayNamespace[Array]
) -> Array:
    r"""Log of the probability density function of the normal distribution.

    Parameters
    ----------
    x : Array
        x at which to evaluate the PDF.
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
    return -0.5 * (((x - loc) / sigma) ** 2 + 2 * xp.log(sigma) + log2pi)


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
        return norm_logpdf(
            data[c], mpars[c, "mu"], self.xp.clip(mpars[c, "sigma"], 1e-10), xp=self.xp
        )


# ============================================================================


log2 = math.log(2)


def log_truncation_term(
    ab: tuple[Array, Array], /, loc: Array, sigma: Array, *, xp: ArrayNamespace[Array]
) -> Array:
    """Log of integral from a to b of normal."""
    erfa = xp.erf((ab[0] - loc) / sigma / sqrt2)  # type: ignore[attr-defined]
    erfb = xp.erf((ab[1] - loc) / sigma / sqrt2)  # type: ignore[attr-defined]
    return xp.log(erfb - erfa) - log2


def truncnorm_logpdf(
    x: Array,
    /,
    loc: Array,
    sigma: Array,
    ab: tuple[float | Array, float | Array],
    *,
    xp: ArrayNamespace[Array],
) -> Array:
    out = xp.full_like(x, -xp.inf)
    sel = (ab[0] <= x) & (x <= ab[1])
    out[sel] = norm_logpdf(
        x[sel], loc=loc[sel], sigma=sigma[sel], xp=xp
    ) - log_truncation_term(ab, loc=loc[sel], sigma=sigma[sel], xp=xp)
    return out


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
        return truncnorm_logpdf(
            data[c],
            loc=mpars[c, "mu"],
            sigma=self.xp.clip(mpars[c, "sigma"], 1e-10),
            ab=self.coord_bounds[self.coord_names[0]],
            xp=self.xp,
        )
