"""Register extensions and single-dispatch functions with stream_ml."""

__all__: list[str] = []

import torch as xp
from torch import nn

from stream_ml.core._core.base import NN_NAMESPACE

# Register the torch array namespace.
NN_NAMESPACE[xp] = nn
