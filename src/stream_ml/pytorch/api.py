"""Core feature."""

from __future__ import annotations

from abc import abstractmethod
from math import inf
from typing import ClassVar, Protocol

from torch import nn

from stream_ml.core.api import Model as CoreModel
from stream_ml.core.data import Data
from stream_ml.pytorch.prior.bounds import PriorBounds, SigmoidBounds
from stream_ml.pytorch.typing import Array
from stream_ml.core.typing import ArrayNamespace

__all__: list[str] = []


class Model(CoreModel[Array], Protocol):
    """Pytorch model base class.

    Parameters
    ----------
    n_features : int
        The number off features used by the NN.

    name : str or None, optional keyword-only
        The (internal) name of the model, e.g. 'stream' or 'background'. Note
        that this can be different from the name of the model when it is used in
        a mixture model (see :class:`~stream_ml.core.core.MixtureModel`).
    """

    DEFAULT_BOUNDS: ClassVar[PriorBounds] = SigmoidBounds(-inf, inf)

    def __post_init__(self, array_namespace: ArrayNamespace) -> None:
        nn.Module.__init__(self)  # Needed for PyTorch
        super().__post_init__(array_namespace=array_namespace)

    # ========================================================================
    # ML

    @abstractmethod
    def forward(self, data: Data[Array]) -> Array:
        """Forward pass.

        Parameters
        ----------
        data : Data[Array]
            Input.

        Returns
        -------
        Array
            fraction, mean, sigma
        """
        raise NotImplementedError
