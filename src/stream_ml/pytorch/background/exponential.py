"""Built-in background models."""

from __future__ import annotations

from dataclasses import KW_ONLY, InitVar, dataclass

import torch as xp
from torch import nn

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
_1 = xp.asarray(1)


@dataclass(unsafe_hash=True)
class Exponential(ModelBase):
    r"""Tilted separately in each dimension.

    In each dimension the background is an exponential distribution between
    points ``a`` and ``b``. The rate parameter is ``m``.

    The non-zero portion of the PDF, where :math:`a < x < b` is

    .. math::

        f(x) = \frac{m * e^{-m * (x -a)}}{1 - e^{-m * (b - a)}}

    However, we use the order-3 Taylor expansion of the exponential function
    around m=0, to avoid the m=0 indeterminancy.

    .. math::

        f(x) = \frac{1}{b-a} + m * (0.5 - \frac{x-a}{b-a}) + \frac{m^2}{2} *
        (\frac{b-a}{6} - (x-a) + \frac{(x-a)^2}{b-a})
    """

    net: InitVar[nn.Module | None] = None

    _: KW_ONLY
    array_namespace: InitVar[ArrayNamespace[Array]]
    param_names: ParamNamesField = ParamNamesField(
        (WEIGHT_NAME, (..., ("slope",))), requires_all_coordinates=False
    )
    param_bounds: ParamBoundsField[Array] = ParamBoundsField[Array](
        {WEIGHT_NAME: SigmoidBounds(_eps, 1.0, param_name=(WEIGHT_NAME,))}
    )
    require_mask: bool = False

    def __post_init__(
        self, array_namespace: ArrayNamespace[Array], net: nn.Module | None
    ) -> None:
        super().__post_init__(array_namespace=array_namespace)

        n_slopes = len(self.param_names) - 1  # (don't count the weight)

        # Initialize the network
        if net is None:
            self.nn = nn.Linear(1, n_slopes)
            # Note; would prefer nn.Parameter(xp.zeros((1, n_slopes)) + 1e-5)
            # as that has 1/2 as many params, but it's not callable.
        else:
            self.nn = net
            # TODO: ensure n_out == n_slopes

        # Pre-compute the associated constant factors
        self._a = xp.asarray(
            [
                a
                for k, (a, _) in self.coord_bounds.items()
                if k in self.param_names.top_level
            ]
        )
        self._b = xp.asarray(
            [
                b
                for k, (_, b) in self.coord_bounds.items()
                if k in self.param_names.top_level
            ]
        )
        self._bma = self._b - self._a

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
        f = mpars[(WEIGHT_NAME,)]
        eps = xp.finfo(f.dtype).eps  # TOOD: or tiny?

        # The mask is used to indicate which data points are available. If the
        # mask is not provided, then all data points are assumed to be
        # available.
        if mask is not None:
            indicator = mask[tuple(self.coord_bounds.keys())].array.int()
        elif self.require_mask:
            msg = "mask is required"
            raise ValueError(msg)
        else:
            indicator = xp.ones_like(f, dtype=xp.int)
            # This has shape (N, 1) so will broadcast correctly.

        # Data
        d_arr = data[self.coord_names].array - self._a
        # Get the slope from `mpars` we check param_names to see if the
        # slope is a parameter. If it is not, then we assume it is 0.
        # When the slope is 0, the log-likelihood reduces to a Uniform.
        ms = xp.hstack(
            tuple(
                mpars[(k, "slope")]
                if (k, "slope") in self.param_names.flats
                else xp.zeros((len(d_arr), 1))
                for k in self.coord_names
            )
        )
        # log-likelihood
        lnliks = xp.log(
            1 / self._bma
            + (ms * (0.5 - d_arr / self._bma))
            + (ms**2 * (self._bma / 6 - d_arr + d_arr**2 / self._bma) / 2)
        )

        return xp.log(xp.clip(f, eps)) + (indicator * lnliks).sum(dim=1, keepdim=True)

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
        pred = (xp.sigmoid(self.nn(data[self.indep_coord_name])) - 0.5) / self._bma
        pred = xp.hstack((xp.zeros((len(pred), 1)), pred))  # add the weight
        return self._forward_priors(pred, data)
