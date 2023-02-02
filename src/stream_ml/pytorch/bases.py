"""Base for multi-component models."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from math import inf
from typing import TYPE_CHECKING, ClassVar

import torch as xp
from torch import nn

from stream_ml.core.api import Model
from stream_ml.core.bases import ModelsBase as CoreModelsBase
from stream_ml.core.prior.base import PriorBase
from stream_ml.core.utils.frozen_dict import FrozenDictField
from stream_ml.pytorch.prior.bounds import PriorBounds, SigmoidBounds
from stream_ml.pytorch.typing import Array

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.core.data import Data

__all__: list[str] = []


@dataclass
class ModelsBase(nn.Module, CoreModelsBase[Array]):
    """Multi-model base class."""

    components: FrozenDictField[str, Model] = FrozenDictField()

    _: KW_ONLY
    priors: tuple[PriorBase[Array], ...] = ()

    DEFAULT_BOUNDS: ClassVar[PriorBounds] = SigmoidBounds(-inf, inf)

    def __post_init__(self) -> None:
        super().__post_init__()

        # Register the models with pytorch.
        nn.Module.__init__(self)  # Needed for PyTorch
        for name, model in self.components.items():
            self.add_module(name=name, module=model)

    # ========================================================================
    # ML

    def forward(self, data: Data[Array], /) -> Array:
        """Forward pass.

        Parameters
        ----------
        data : Data
            Input.

        Returns
        -------
        Array
            fraction, mean, sigma.
        """
        nn = xp.concat([model(data) for model in self.components.values()], dim=1)

        # Call the prior to limit the range of the parameters
        # TODO: a better way to do the order of the priors.
        for prior in self.priors:
            nn = prior(nn, data, self)

        return nn
