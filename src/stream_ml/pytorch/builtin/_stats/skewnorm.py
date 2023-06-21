from __future__ import annotations

__all__: list[str] = []

from typing import TYPE_CHECKING

import torch as xp

from stream_ml.core.builtin._stats.norm import cdf as norm_cdf
from stream_ml.core.builtin._stats.norm import log2
from stream_ml.core.builtin._stats.norm import logcdf as norm_logcdf
from stream_ml.core.builtin._stats.norm import logpdf as norm_logpdf

if TYPE_CHECKING:
    from stream_ml.pytorch.typing import Array


_t_arr = xp.linspace(0, 1, 1_000)


def _owens_t_integrand(x: Array, t: Array) -> Array:
    return xp.exp(-0.5 * x**2 * (1 + t**2)) / (1 + t**2)


def owens_t_approx(x: Array, a: Array) -> Array:
    # https://en.wikipedia.org/wiki/Owen%27s_T_function
    t = _t_arr.repeat(len(a), 1) * a[:, None]  # (N, 1000)
    return (
        xp.sum(_owens_t_integrand(x[:, None], t), axis=-1)
        * (t[:, 1] - t[:, 0])
        / (2 * xp.pi)
    )


def logpdf(x: Array, /, loc: Array, sigma: Array, skew: Array) -> Array:
    # https://en.wikipedia.org/wiki/Skew_normal_distribution
    return (
        log2
        + norm_logpdf(x, loc=loc, sigma=sigma, xp=xp)
        + norm_logcdf(x, loc=loc, sigma=sigma / skew, xp=xp)
    )


def cdf(x: Array | float, /, loc: Array, sigma: Array, skew: Array) -> Array:
    return norm_cdf(x, loc, sigma, xp=xp) - 2 * owens_t_approx((x - loc) / sigma, skew)


def logcdf(x: Array, /, loc: Array, sigma: Array, skew: Array) -> Array:
    return xp.log(cdf(x, loc, sigma, skew))
