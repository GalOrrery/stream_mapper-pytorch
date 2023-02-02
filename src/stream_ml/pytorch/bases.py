"""Base for multi-component models."""

from __future__ import annotations

# STDLIB
from abc import abstractmethod
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

# THIRD-PARTY
import torch as xp
from torch import nn

# LOCAL
from stream_ml.core.bases import ModelsBase as CoreModelsBase
from stream_ml.core.prior.base import PriorBase
from stream_ml.core.utils.frozen_dict import FrozenDictField
from stream_ml.pytorch.api import Model
from stream_ml.pytorch.typing import Array

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.core.data import Data
    from stream_ml.core.params import Params

__all__: list[str] = []


@dataclass
class ModelsBase(nn.Module, CoreModelsBase[Array], Model):  # type: ignore[misc]
    """Multi-model base class."""

    components: FrozenDictField[str, Model] = FrozenDictField()  # type: ignore[assignment]  # noqa: E501

    _: KW_ONLY
    priors: tuple[PriorBase[Array], ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()

        # Register the models with pytorch.
        for name, model in self.components.items():
            self.add_module(name=name, module=model)

    # ===============================================================
    # Mapping

    def __getitem__(self, key: str) -> Model:
        return self.components[key]

    # ===============================================================
    # Statistics

    @abstractmethod
    def ln_likelihood_arr(
        self, mpars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Log likelihood."""
        raise NotImplementedError

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
