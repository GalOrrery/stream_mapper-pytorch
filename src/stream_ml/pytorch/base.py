"""Core feature."""

from __future__ import annotations

from dataclasses import InitVar, dataclass
from math import inf
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from torch import nn

from stream_ml.core.base import ModelBase as CoreModelBase
from stream_ml.core.prior.bounds import PriorBounds  # noqa: TCH001
from stream_ml.pytorch.prior.bounds import SigmoidBounds
from stream_ml.pytorch.typing import Array

__all__: list[str] = []


if TYPE_CHECKING:
    from stream_ml.core.data import Data
    from stream_ml.core.typing import ArrayNamespace

    Self = TypeVar("Self", bound="ModelBase")


@dataclass(unsafe_hash=True)
class ModelBase(nn.Module, CoreModelBase[Array]):
    """Model base class."""

    net: InitVar[nn.Module | None] = None

    DEFAULT_BOUNDS: ClassVar[PriorBounds] = SigmoidBounds(-inf, inf)

    def __new__(  # noqa: D102
        cls: type[Self], *args: Any, **kwargs: Any  # noqa: ARG003
    ) -> Self:
        self = object.__new__(cls)

        # PyTorch needs to be initialized before attributes are assigned.
        nn.Module.__init__(self)
        return self

    def __post_init__(
        self, array_namespace: ArrayNamespace[Array], net: nn.Module | None
    ) -> None:
        super().__post_init__(array_namespace=array_namespace)

        # Need to type hint the nn.Module
        self.nn: nn.Module
        if net is not None:
            self.nn = net
        else:
            msg = "must provide a wrapped neural network."
            raise ValueError(msg)

    # ========================================================================
    # ML

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
        return self._forward_priors(self.nn(data[self.indep_coord_name]), data)
