"""Track priors."""

from __future__ import annotations

__all__ = [
    # from stream_ml.core.prior
    "Prior",
    "FunctionPrior",
    "HardThreshold",
    # from here
    "ControlRegions",
]

from dataclasses import field, make_dataclass

from stream_ml.core.prior import ControlRegions as CoreControlRegions
from stream_ml.core.prior import FunctionPrior, Prior
from stream_ml.core.prior import HardThreshold as CoreHardThreshold
from stream_ml.core.typing import ArrayNamespace

from stream_ml.pytorch.typing import Array

HardThreshold = make_dataclass(
    "HardThreshold",
    [("array_namespace", ArrayNamespace[Array], field(default="torch", kw_only=True))],
    bases=(CoreHardThreshold[Array],),
    frozen=True,
    repr=False,
    unsafe_hash=True,
)


ControlRegions = make_dataclass(
    "ControlRegions",
    [("array_namespace", ArrayNamespace[Array], field(default="torch", kw_only=True))],
    bases=(CoreControlRegions[Array],),
    frozen=True,
    repr=False,
    unsafe_hash=True,
)
