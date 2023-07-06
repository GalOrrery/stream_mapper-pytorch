"""Register extensions and single-dispatch functions with stream_ml."""

__all__: list[str] = []

from stream_ml.pytorch._connect import compat, data, funcs, nn_namespace  # noqa: F401
