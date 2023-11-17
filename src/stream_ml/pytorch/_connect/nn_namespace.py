"""Register extensions and single-dispatch functions with stream_ml."""

__all__: tuple[str, ...] = ()

from torch import nn
import torch as xp

from stream_ml.core._connect.nn_namespace import NN_NAMESPACE

# Register the torch array namespace.
NN_NAMESPACE[xp] = nn
