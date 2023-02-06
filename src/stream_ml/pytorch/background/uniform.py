"""Built-in background models."""

from __future__ import annotations

from dataclasses import KW_ONLY, InitVar, dataclass

import torch as xp

from stream_ml.core.data import Data
from stream_ml.core.params import Params
from stream_ml.core.params.bounds import ParamBoundsField
from stream_ml.core.params.names import ParamNamesField
from stream_ml.core.setup_package import WEIGHT_NAME
from stream_ml.core.typing import ArrayNamespace
from stream_ml.pytorch.base import ModelBase
from stream_ml.pytorch.prior.bounds import SigmoidBounds
from stream_ml.pytorch.typing import Array

__all__: list[str] = []


_eps = float(xp.finfo(xp.float32).eps)


@dataclass(unsafe_hash=True)
class Uniform(ModelBase):
    """Uniform background model."""

    _: KW_ONLY
    array_namespace: InitVar[ArrayNamespace[Array]]
    param_names: ParamNamesField = ParamNamesField((WEIGHT_NAME,))
    param_bounds: ParamBoundsField[Array] = ParamBoundsField[Array](
        {WEIGHT_NAME: SigmoidBounds(_eps, 1.0, param_name=(WEIGHT_NAME,))}
    )
    require_mask: bool = False

    def __post_init__(self, array_namespace: ArrayNamespace[Array]) -> None:
        super().__post_init__(array_namespace=array_namespace)

        # Pre-compute the log-difference, shape (1, F)
        self._ln_diffs = xp.log(
            xp.asarray([b - a for a, b in self.coord_bounds.values()])[None, :]
        )

    # ========================================================================
    # Statistics

    def ln_likelihood_arr(
        self,
        mpars: Params[Array],
        data: Data[Array],
        *,
        mask: Data[Array] | None = None,
        **kwargs: Array,
    ) -> Array:
        """Log-likelihood of the background.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array]
            Labelled data.

        mask : (N, 1) Data[Array[bool]], keyword-only
            Data availability. `True` if data is available, `False` if not.
            Should have the same keys as `data`.
        **kwargs : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        f = mpars[(WEIGHT_NAME,)]
        eps = xp.finfo(f.dtype).eps  # TOOD: or tiny?

        if mask is not None:
            indicator = mask[tuple(self.coord_bounds.keys())].array.int()
        elif self.require_mask:
            msg = "mask is required"
            raise ValueError(msg)
        else:
            indicator = xp.ones_like(self._ln_diffs, dtype=xp.int)
            # shape (1, F) so that it can broadcast with (N, F)

        return xp.log(xp.clip(f, eps)) - (indicator * self._ln_diffs).sum(
            dim=1, keepdim=True
        )

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
        return xp.asarray([])  # there are no priors
