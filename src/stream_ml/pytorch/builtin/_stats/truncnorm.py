from __future__ import annotations

__all__: list[str] = []

from typing import TYPE_CHECKING

import torch as xp

from stream_ml.core.builtin._stats.norm import cdf as norm_cdf
from stream_ml.core.builtin._stats.norm import logpdf as norm_logpdf

if TYPE_CHECKING:
    from stream_ml.pytorch.typing import Array


def _logpdf(
    x: Array, /, loc: Array, sigma: Array, a: Array | float, b: Array | float
) -> Array:
    log_trunc = xp.log(norm_cdf(b, loc, sigma) - norm_cdf(a, loc, sigma))
    return norm_logpdf(x, loc=loc, sigma=sigma) - log_trunc


def logpdf(
    x: Array,
    /,
    loc: Array,
    sigma: Array,
    a: Array | float,
    b: Array | float,
    *,
    nil: float = -xp.inf,
) -> Array:
    out = xp.full_like(x, nil)
    sel = (a <= x) & (x <= b)
    out[sel] = _logpdf(x[sel], loc=loc[sel], sigma=sigma[sel], a=a, b=b)
    return out
