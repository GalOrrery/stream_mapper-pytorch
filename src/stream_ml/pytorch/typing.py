"""Pytorch type hints."""

__all__ = ["Array", "NNModel", "ArrayNamespace"]

from torch import Tensor as Array
from torch.nn import Module as NNModel

from stream_ml.core.typing import ArrayNamespace
