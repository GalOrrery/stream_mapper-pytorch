"""Built-in background models."""

from __future__ import annotations

__all__: list[str] = [
    # Mass Function
    "StreamMassFunction",
    "UniformStreamMassFunction",
    "HardCutoffMassFunction",
    "StepwiseMassFunction",
    # Core
    "IsochroneMVNorm",
    # Utils
    "Parallax2DistMod",
]

from dataclasses import field, make_dataclass

import torch as xp

from stream_ml.core.builtin._isochrone.mf import (
    HardCutoffMassFunction,
    StepwiseMassFunction,
    StreamMassFunction,
    UniformStreamMassFunction,
)
from stream_ml.core.builtin._isochrone.utils import (
    Parallax2DistMod as CoreParallax2DistMod,
)

from stream_ml.pytorch.builtin._isochrone.core import IsochroneMVNorm
from stream_ml.pytorch.typing import Array, ArrayNamespace

# -----------------------------------------------------------------------------

Parallax2DistMod = make_dataclass(
    "Parallax2DistMod",
    [("array_namespace", ArrayNamespace[Array], field(default=xp, kw_only=True))],
    bases=(CoreParallax2DistMod[Array],),
    unsafe_hash=True,
    repr=False,
)
