"""Parameters."""

__all__: list[str] = [
    # modules
    "bounds",
    "scaler",
    # parameters
    "ModelParameter",
    "ModelParameters",
    "ModelParametersField",
    # values
    "Params",
    "freeze_params",
    "unfreeze_params",
    "set_param",
    "add_prefix",
]

from stream_mapper.core.params import (
    ModelParameter,
    ModelParameters,
    ModelParametersField,
    Params,
    add_prefix,
    freeze_params,
    set_param,
    unfreeze_params,
)

from stream_mapper.pytorch.params import bounds, scaler
