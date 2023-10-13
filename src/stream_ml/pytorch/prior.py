"""Track priors."""

from __future__ import annotations

from dataclasses import field, make_dataclass

from stream_ml.core import prior
from stream_ml.core.prior import *  # noqa: F403
from stream_ml.core.prior._track import ControlRegions as CoreControlRegions
from stream_ml.core.prior._weight import HardThreshold as CoreHardThreshold
from stream_ml.core.typing import ArrayNamespace

from stream_ml.pytorch.typing import Array

__all__ = prior.__all__

HardThreshold = make_dataclass(
    "HardThreshold",
    [("array_namespace", ArrayNamespace[Array], field(default="torch", kw_only=True))],
    bases=(CoreHardThreshold[Array],),
    frozen=True,
    repr=False,
)


ControlRegions = make_dataclass(
    "ControlRegions",
    [("array_namespace", ArrayNamespace[Array], field(default="torch", kw_only=True))],
    bases=(CoreControlRegions[Array],),
    frozen=True,
    repr=False,
)
