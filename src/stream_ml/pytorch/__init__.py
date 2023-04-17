"""Stream Memberships Likelihood, with ML."""

from stream_ml.core.data import Data
from stream_ml.pytorch import builtin, compat, nn, params, prior, utils
from stream_ml.pytorch._base import ModelBase
from stream_ml.pytorch.multi import IndependentModels, MixtureModel

__all__ = [
    # classes
    "Data",
    # modules
    "builtin",
    "compat",
    "nn",
    "params",
    "prior",
    "utils",
    # model classes
    "ModelBase",
    "MixtureModel",
    "IndependentModels",
]

# Register with single-dispatch
from stream_ml.pytorch import _connect  # noqa: F401
