from __future__ import annotations

__all__: list[str] = []

import math
from typing import TYPE_CHECKING

import torch as xp

if TYPE_CHECKING:
    from stream_ml.pytorch.typing import Array


sqrt2 = math.sqrt(2)
log2 = math.log(2)
log2pi = math.log(2 * math.pi)


def logpdf(x: Array | float, loc: Array, sigma: Array) -> Array:
    return -0.5 * (((x - loc) / sigma) ** 2 + 2 * xp.log(sigma) + log2pi)


def cdf(x: Array | float, loc: Array, sigma: Array) -> Array:
    return 0.5 * xp.erfc((loc - x) / sigma / sqrt2)


def logcdf(x: Array | float, loc: Array, sigma: Array) -> Array:
    return xp.log(cdf(x, loc, sigma))
