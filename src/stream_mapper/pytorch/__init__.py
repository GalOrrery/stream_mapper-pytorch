"""Stream Memberships Likelihood, with ML."""

from stream_mapper.core import Data, Params

from stream_mapper.pytorch import builtin, nn, params, prior, utils
from stream_mapper.pytorch._base import ModelBase
from stream_mapper.pytorch._multi import IndependentModels, MixtureModel

__all__ = (
    # modules
    "builtin",
    "nn",
    "params",
    "prior",
    "utils",
    # model classes
    "ModelBase",
    "MixtureModel",
    "IndependentModels",
    # classes
    "Data",
    "Params",
)

# Register with single-dispatch
from stream_mapper.pytorch import _connect  # noqa: F401
