"""Core feature."""

from __future__ import annotations

__all__: list[str] = []

from contextlib import nullcontext
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

import torch as xp

from stream_ml.core.params.scaler import scale_params

from stream_ml.pytorch._base import ModelBase
from stream_ml.pytorch.utils import names_intersect

if TYPE_CHECKING:
    from stream_ml.pytorch import Data
    from stream_ml.pytorch.params import Params
    from stream_ml.pytorch.typing import Array


@dataclass(unsafe_hash=True)
class FlowModel(ModelBase):
    """Normalizing flow model."""

    _: KW_ONLY
    jacobian_logdet: float  # Log of the Jacobian determinant
    with_grad: bool = True

    def ln_likelihood(
        self, mpars: Params[Array], /, data: Data[Array], **kwargs: Array
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
        data = self.data_scaler.transform(
            data, names=names_intersect(data, self.data_scaler), xp=self.xp
        )
        mpars = scale_params(self, mpars)

        with nullcontext() if self.with_grad else xp.no_grad():
            return self.jacobian_logdet + self.net.log_prob(
                inputs=data[self.coord_names].array,
                context=data[self.indep_coord_names].array
                if self.indep_coord_names is not None
                else None,
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
