"""Stream Memberships Likelihood, with ML."""

from dataclasses import field, make_dataclass

import torch as xp

from stream_ml.core.prior import FunctionPrior, Prior
from stream_ml.core.prior import HardThreshold as CoreHardThreshold

from stream_ml.pytorch.prior._track import ControlPoints, ControlRegions
from stream_ml.pytorch.typing import Array, ArrayNamespace

__all__ = [
    # from stream_ml.core.prior
    "Prior",
    "FunctionPrior",
    "HardThreshold",
    # from here
    "ControlPoints",
    "ControlRegions",
]


HardThreshold = make_dataclass(
    "HardThreshold",
    [("array_namespace", ArrayNamespace[Array], field(default=xp, kw_only=True))],
    bases=(CoreHardThreshold[Array],),
    frozen=True,
    unsafe_hash=True,
)
