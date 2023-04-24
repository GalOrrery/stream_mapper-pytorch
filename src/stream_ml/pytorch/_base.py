"""Core feature."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

import torch as xp
from torch import nn

from stream_ml.core import ModelBase as CoreModelBase
from stream_ml.core.prior.bounds import NoBounds
from stream_ml.pytorch.typing import Array, NNModel

__all__: list[str] = []


if TYPE_CHECKING:
    from stream_ml.core.base import NNField
    from stream_ml.core.data import Data
    from stream_ml.core.prior.bounds import PriorBounds
    from stream_ml.core.typing import ArrayNamespace

    Self = TypeVar("Self", bound="ModelBase")


@dataclass(unsafe_hash=True)
class ModelBase(nn.Module, CoreModelBase[Array, NNModel]):
    """Model base class."""

    DEFAULT_PARAM_BOUNDS: ClassVar[PriorBounds] = NoBounds()

    _: KW_ONLY
    array_namespace: ArrayNamespace[Array] = xp

    def __new__(cls: type[Self], *args: Any, **kwargs: Any) -> Self:
        self: Self = super().__new__(cls, *args, **kwargs)  # <- CoreModelBase

        # PyTorch needs to be initialized before attributes are assigned.
        nn.Module.__init__(self)
        return self

    def __post_init__(self) -> None:
        super().__post_init__()

        # Net needs to added to ensure that it's registered as a module.
        # TODO! not need to overwrite the descriptor.
        self.net: NNField[NNModel] = self.net

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
        if self.net is None:
            return self.xp.asarray([])
        # The forward step runs on the normalized coordinates
        scaled_data = self.data_scaler.transform(data, names=data.names)
        return self._forward_priors(
            self.net(scaled_data[:, self.indep_coord_names, 0]), scaled_data
        )
