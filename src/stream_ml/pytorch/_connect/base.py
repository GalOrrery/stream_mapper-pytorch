"""Register extensions and single-dispatch functions with stream_ml."""

import torch as xp
from torch import nn

from stream_ml.core.base import NN_NAMESPACE

__all__: list[str] = []

# Register the torch array namespace.
NN_NAMESPACE[xp] = nn
