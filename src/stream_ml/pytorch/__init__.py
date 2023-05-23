"""Stream Memberships Likelihood, with ML."""

from stream_ml.core.data import Data

from stream_ml.pytorch import builtin, nn, params, prior, utils
from stream_ml.pytorch._base import ModelBase
from stream_ml.pytorch._multi import IndependentModels, MixtureModel

__all__ = [
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
]

# Register with single-dispatch
from stream_ml.pytorch import _connect  # noqa: F401
