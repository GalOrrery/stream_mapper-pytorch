"""Core feature."""

from __future__ import annotations

from collections.abc import Callable  # noqa: TCH003
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

import torch as xp

from stream_ml.pytorch._base import ModelBase

__all__: list[str] = []


if TYPE_CHECKING:
    from stream_ml.core.data import Data
    from stream_ml.core.params import Params

    from stream_ml.pytorch.typing import Array


@dataclass(unsafe_hash=True)
class KDEModel(ModelBase):
    """Kernel Density Estimate model."""

    _: KW_ONLY
    kernel: Callable[[Array], Array]

    transpose: bool
    include_indep_coords: bool

    def __post_init__(self) -> None:
        """Post-initialization hook."""
        if self.net is not None:
            msg = "Cannot pass `net` to KDEModel."
            raise ValueError(msg)

        self._all_coord_names: tuple[str, ...]
        object.__setattr__(
            self,
            "_all_coord_names",
            (self.indep_coord_names if self.include_indep_coords else ())
            + self.coord_names,
        )

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
            Data.

        **kwargs : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        with xp.no_grad():
            return xp.log(
                xp.clip(
                    xp.asarray(self.kernel(data[:, self._all_coord_names, 0])), min=0
                )
            )[:, None]

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
