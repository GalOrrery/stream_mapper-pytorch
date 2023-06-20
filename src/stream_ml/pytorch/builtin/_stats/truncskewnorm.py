from __future__ import annotations

__all__: list[str] = []

from typing import TYPE_CHECKING

import torch as xp

from stream_ml.pytorch.builtin._stats.skewnorm import cdf as skewnorm_cdf
from stream_ml.pytorch.builtin._stats.skewnorm import logpdf as skewnorm_logpdf

if TYPE_CHECKING:
    from stream_ml.pytorch.typing import Array


def _logpdf(
    x: Array,
    /,
    loc: Array,
    sigma: Array,
    skew: Array,
    a: Array | float,
    b: Array | float,
) -> Array:
    log_trunc = xp.log(
        skewnorm_cdf(b, loc, sigma, skew) - skewnorm_cdf(a, loc, sigma, skew)
    )
    return skewnorm_logpdf(x, loc=loc, sigma=sigma, skew=skew) - log_trunc


def logpdf(  # noqa: PLR0913
    x: Array,
    /,
    loc: Array,
    sigma: Array,
    skew: Array,
    a: Array | float,
    b: Array | float,
    *,
    nil: float = -xp.inf,
) -> Array:
    out = xp.full_like(x, nil)
    sel = (a <= x) & (x <= b)
    out[sel] = xp.clip(
        _logpdf(x[sel], loc=loc[sel], sigma=sigma[sel], skew=skew[sel], a=a, b=b),
        nil=nil,
    )
    return out
