"""Core feature."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

import torch as xp

from stream_ml.core.setup_package import WEIGHT_NAME
from stream_ml.pytorch.base import ModelBase

__all__: list[str] = []


if TYPE_CHECKING:
    from stream_ml.core.data import Data
    from stream_ml.core.params import Params
    from stream_ml.core.typing import ArrayNamespace
    from stream_ml.pytorch.typing import Array


@dataclass(unsafe_hash=True)
class FlowModel(ModelBase):
    """Normalizing flow model."""

    # net: NNField[Flow] = NNField[Flow](default=None)  # noqa: ERA001

    _: KW_ONLY
    with_grad: bool = True
    context_coord_names: tuple[str, ...] | None = None

    def __post_init__(self, array_namespace: ArrayNamespace[Array]) -> None:
        super().__post_init__(array_namespace=array_namespace)

    def ln_likelihood_arr(
        self,
        mpars: Params[Array],
        data: Data[Array],
        **kwargs: Array,
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
        ln_weight = (
            self.xp.log(mpars[(WEIGHT_NAME,)])
            if WEIGHT_NAME in mpars
            else self.xp.asarray(0)
        )

        if not self.with_grad:
            with xp.no_grad():
                return (
                    ln_weight
                    + self.nn.log_prob(
                        inputs=data[:, self.coord_names, 0],
                        context=data[self.context_coord_names].array
                        if self.context_coord_names is not None
                        else None,
                    )[:, None]
                )

        return (
            ln_weight
            + self.nn.log_prob(
                inputs=data[:, self.coord_names, 0],
                context=data[self.context_coord_names].array
                if self.context_coord_names is not None
                else None,
            )[:, None]
        )

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
