"""Stream Memberships Likelihood, with ML."""

from stream_ml.core.data import Data
from stream_ml.pytorch import background, nn, params, prior, stream, utils
from stream_ml.pytorch.multi import IndependentModels, MixtureModel

__all__ = [
    # classes
    "Data",
    # modules
    "background",
    "stream",
    "utils",
    "params",
    "prior",
    "nn",
    # model classes
    "MixtureModel",
    "IndependentModels",
]

# Register with single-dispatch
from stream_ml.pytorch import _connect  # noqa: F401
