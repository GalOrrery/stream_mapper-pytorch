"""Track priors."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field

from stream_ml.core import prior
from stream_ml.core.prior import *  # noqa: F403
from stream_ml.core.prior._track import ControlRegions as CoreControlRegions
from stream_ml.core.prior._weight import HardThreshold as CoreHardThreshold
from stream_ml.core.typing import ArrayNamespace

from stream_ml.pytorch.typing import Array

__all__ = prior.__all__


@dataclass(frozen=True, repr=False)
class HardThreshold(CoreHardThreshold[Array]):
    _: KW_ONLY
    array_namespace: ArrayNamespace[Array] = field(default="torch", kw_only=True)  # type: ignore[arg-type]


@dataclass(frozen=True, repr=False)
class ControlRegions(CoreControlRegions[Array]):
    _: KW_ONLY
    array_namespace: ArrayNamespace[Array] = field(default="torch", kw_only=True)  # type: ignore[arg-type]
