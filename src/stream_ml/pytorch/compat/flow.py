"""Core feature."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

import torch as xp

from stream_ml.core.params.names import ParamNamesField
from stream_ml.core.setup_package import WEIGHT_NAME
from stream_ml.core.utils.scale.utils import scale_params
from stream_ml.pytorch.base import ModelBase

__all__: list[str] = []


if TYPE_CHECKING:
    from stream_ml.core.data import Data
    from stream_ml.core.params import Params
    from stream_ml.pytorch.typing import Array


@dataclass(unsafe_hash=True)
class FlowModel(ModelBase):
    """Normalizing flow model."""

    # net: NNField[Flow] = NNField[Flow](default=None)

    _: KW_ONLY
    with_grad: bool = True
    context_coord_names: tuple[str, ...] | None = None
    param_names: ParamNamesField = ParamNamesField((WEIGHT_NAME,))

    def ln_likelihood(
        self, mpars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Log-likelihood of the array.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters. The flow has an internal weight, so we don't use the
            weight, if passed.
        data : Data[Array]
            Data (phi1, phi2).

        **kwargs : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        data = self.data_scaler.transform(data, names=self.data_scaler.names)
        mpars = scale_params(self, mpars)

        # ln_weight = (
        #     self.xp.log(mpars[(WEIGHT_NAME,)])
        #     if WEIGHT_NAME in mpars
        #     else self.xp.asarray(0)
        # )
        ln_wgt = self.xp.asarray(0)  # TODO! have some way to turn on the weight.

        if not self.with_grad:
            with xp.no_grad():
                return (
                    ln_wgt
                    + self.net.log_prob(
                        inputs=data[:, self.coord_names, 0],
                        context=data[:, self.context_coord_names, 0]
                        if self.context_coord_names is not None
                        else None,
                    )[:, None]
                )

        return (
            ln_wgt
            + self.net.log_prob(
                inputs=data[:, self.coord_names, 0],
                context=data[:, self.context_coord_names, 0]
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
        return xp.ones((len(data), 1))  # the weight
