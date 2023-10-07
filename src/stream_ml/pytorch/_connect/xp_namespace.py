"""Register extensions and single-dispatch functions with stream_ml."""

__all__: list[str] = []

import torch as xp

from stream_ml.core._connect.xp_namespace import XP_NAMESPACE, XP_NAMESPACE_REVERSE

# Register the torch array namespace.
XP_NAMESPACE[xp] = xp
XP_NAMESPACE["torch"] = xp
XP_NAMESPACE["pytorch"] = xp

XP_NAMESPACE_REVERSE[xp] = "torch"
