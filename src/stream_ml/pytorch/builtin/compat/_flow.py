"""Core feature."""

from __future__ import annotations

from abc import abstractmethod

__all__: list[str] = []

from contextlib import nullcontext
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

import torch as xp

from stream_ml.core.builtin._utils import WhereRequiredError

from stream_ml.pytorch._base import ModelBase
from stream_ml.pytorch.utils import names_intersect

if TYPE_CHECKING:
    from stream_ml.pytorch import Data
    from stream_ml.pytorch.params import Params
    from stream_ml.pytorch.typing import Array


@dataclass(unsafe_hash=True)
class _FlowModel(ModelBase):
    """Normalizing flow model."""

    _: KW_ONLY
    jacobian_logdet: float  # Log of the Jacobian determinant
    with_grad: bool = True

    @abstractmethod
    def _log_prob(self, data: Data[Array], idx: Array) -> Array:
        """Log-probability of the array."""

    def ln_likelihood(
        self,
        mpars: Params[Array],
        /,
        data: Data[Array],
        *,
        where: Data[Array] | None = None,
        **kwargs: Array,
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

        where : Data[Array[(N,), bool]] | None, optional keyword-only
            Where to evaluate the log-likelihood. If not provided, then the
            log-likelihood is evaluated at all data points. ``where`` must
            contain the fields in ``coord_names``. Each field must be a boolean
            array of the same length as `data`. `True` indicates that the data
            point is available, and `False` indicates that the data point is not
            available.

        **kwargs : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        # 'where' is used to indicate which data points are available. If
        # 'where' is not provided, then all data points are assumed to be
        # available.
        where_: Array  # (N, F)
        if where is not None:
            where_ = where[self.coord_names].array
        elif self.require_where:
            raise WhereRequiredError
        else:
            where_ = self.xp.ones((len(data), self.ndim), dtype=bool)
        idx = where_.all(axis=1)
        # TODO: allow for missing data in only some of the dimensions

        data = self.data_scaler.transform(
            data, names=names_intersect(data, self.data_scaler), xp=self.xp
        )

        out = self.xp.zeros(len(data), dtype=data.dtype)
        with nullcontext() if self.with_grad else xp.no_grad():
            out[idx] = self.jacobian_logdet + self._log_prob(data, idx)
        return out

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
