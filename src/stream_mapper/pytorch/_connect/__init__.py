"""Register extensions and single-dispatch functions with stream_mapper."""

__all__: tuple[str, ...] = ()

from stream_mapper.pytorch._connect import (  # noqa: F401
    compat,
    data,
    funcs,
    nn_namespace,
    scaler,
    xp_namespace,
)
