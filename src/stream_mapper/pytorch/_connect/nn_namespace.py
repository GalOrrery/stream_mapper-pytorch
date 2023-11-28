"""Register extensions and single-dispatch functions with stream_mapper."""

__all__: tuple[str, ...] = ()

import torch as xp
from torch import nn

from stream_mapper.core._connect.nn_namespace import NN_NAMESPACE

# Register the torch array namespace.
NN_NAMESPACE[xp] = nn
