"""Built-in background models."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

from torch import nn

from stream_ml.core.params.bounds import ParamBoundsField
from stream_ml.core.params.names import ParamNamesField
from stream_ml.core.setup_package import WEIGHT_NAME
from stream_ml.core.typing import ArrayNamespace  # noqa: TCH001
from stream_ml.pytorch.base import ModelBase
from stream_ml.pytorch.prior.bounds import SigmoidBounds
from stream_ml.pytorch.typing import Array

__all__: list[str] = []

if TYPE_CHECKING:
    from stream_ml.core.data import Data
    from stream_ml.core.params import Params


@dataclass(unsafe_hash=True)
class Sloped(ModelBase):
    r"""Tilted separately in each dimension.

    In each dimension the background is a sloped straight line between points
    ``a`` and ``b``. The slope is ``m``.

    The non-zero portion of the PDF, where :math:`a < x < b` is

    .. math::

        f(x) = m(x - \frac{a + b}{2}) + \frac{1}{b-a}
    """

    _: KW_ONLY
    param_names: ParamNamesField = ParamNamesField(
        (WEIGHT_NAME, (..., ("slope",))), requires_all_coordinates=False
    )
    param_bounds: ParamBoundsField[Array] = ParamBoundsField[Array](
        {WEIGHT_NAME: SigmoidBounds(1e-10, 1.0, param_name=(WEIGHT_NAME,))}
    )
    require_mask: bool = False

    def __post_init__(
        self, array_namespace: ArrayNamespace[Array], net: nn.Module | None
    ) -> None:
        # Initialize the network
        # Note; would prefer nn.Parameter(xp.zeros((1, n_slopes)) + 1e-5)
        # as that has 1/2 as many params, but it's not callable.
        nnet = nn.Linear(1, len(self.coord_bounds)) if net is None else net
        # TODO: ensure n_out == n_slopes
        super().__post_init__(array_namespace=array_namespace, net=nnet)

        # Pre-compute the associated constant factors
        self._bma = self.xp.asarray(
            [
                (b - a)
                for k, (a, b) in self.coord_bounds.items()
                if k in self.param_names.top_level
            ]
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
            Data availability. True if data is available, False if not.
            Should have the same keys as `data`.
        **kwargs : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        wgt = mpars[(WEIGHT_NAME,)]
        eps = self.xp.finfo(wgt.dtype).eps  # TOOD: or tiny?

        # The mask is used to indicate which data points are available. If the
        # mask is not provided, then all data points are assumed to be
        # available.
        if mask is not None:
            indicator = mask[tuple(self.coord_bounds.keys())].array.int()
        elif self.require_mask:
            msg = "mask is required"
            raise ValueError(msg)
        else:
            indicator = self.xp.ones_like(wgt, dtype=self.xp.int)
            # This has shape (N, 1) so will broadcast correctly.

        # Compute the log-likelihood, columns are coordinates.
        lnliks = self.xp.zeros((len(wgt), 4))
        for i, (k, b) in enumerate(self.coord_bounds.items()):
            # Get the slope from `mpars` we check param_names to see if the
            # slope is a parameter. If it is not, then we assume it is 0.
            # When the slope is 0, the log-likelihood reduces to a Uniform.
            m = mpars[(k, "slope")] if (k, "slope") in self.param_names.flats else 0
            lnliks[:, i] = (
                self.xp.log(m * (data[k] - (b[0] + b[1]) / 2) + 1 / (b[1] - b[0]))
            )[:, 0]

        return self.xp.log(self.xp.clip(wgt, eps)) + (indicator * lnliks).sum(
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
        pred = (self.xp.sigmoid(self.nn(data[self.indep_coord_name])) - 0.5) / self._bma
        pred = self.xp.hstack((self.xp.zeros((len(pred), 1)), pred))  # add the weight
        return self._forward_priors(pred, data)
