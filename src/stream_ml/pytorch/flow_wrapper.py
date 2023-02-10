"""Core feature."""

from __future__ import annotations

from dataclasses import KW_ONLY, InitVar, dataclass
from typing import TYPE_CHECKING

import torch as xp
from nflows.flows.base import Flow  # noqa: TCH002

from stream_ml.pytorch.base import ModelBase

__all__: list[str] = []


if TYPE_CHECKING:
    from stream_ml.core.data import Data
    from stream_ml.core.params import Params
    from stream_ml.pytorch.typing import Array


@dataclass(unsafe_hash=True)
class FlowModel(ModelBase):
    """Normalizing flow model."""

    model: InitVar[Flow]
    _: KW_ONLY
    with_grad: bool = True

    def __post_init__(self, model: Flow) -> None:
        super().__post_init__()
        self.wrapped = model

    def ln_likelihood_arr(
        self, mpars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Log-likelihood of the array.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array]
            Data (phi1, phi2).
        **kwargs : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        if not self.with_grad:
            with xp.no_grad():
                return self.wrapped.log_prob(data[self.coord_names].array)[:, None]

        return self.wrapped.log_prob(data[self.coord_names].array)[:, None]

    def forward(self, data: Data[Array]) -> Array:
        """Forward pass.

        Parameters
        ----------
        data : Data[Array]
            Input. Only uses the first argument.

        Returns
        -------
        Array
        """
        return xp.asarray([])
