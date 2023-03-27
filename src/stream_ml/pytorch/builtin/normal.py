"""Gaussian stream model."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
import math
from typing import TYPE_CHECKING

import torch as xp

from stream_ml.core.params.bounds import ParamBoundsField
from stream_ml.core.params.names import ParamNamesField
from stream_ml.core.setup_package import WEIGHT_NAME
from stream_ml.pytorch.base import ModelBase
from stream_ml.pytorch.prior.bounds import SigmoidBounds
from stream_ml.pytorch.typing import Array, NNModel

if TYPE_CHECKING:
    from stream_ml.core.data import Data
    from stream_ml.core.params import Params
    from stream_ml.core.typing import ArrayNamespace

__all__: list[str] = []


_logsqrt2pi = math.log(math.sqrt(2 * math.pi))


def norm_logpdf(value: Array, loc: Array, sigma: Array, *, xp: ArrayNamespace) -> Array:
    r"""Log of the probability density function of the normal distribution.

    Parameters
    ----------
    value : Array
        Value at which to evaluate the PDF.
    loc : Array
        Mean of the distribution.
    sigma : Array
        variance of the distribution.
    xp : ArrayNamespace
        Array namespace.

    Returns
    -------
    Array
        Log of the PDF.
    """
    return -0.5 * ((value - loc) / sigma) ** 2 - xp.log(xp.abs(sigma)) - _logsqrt2pi


@dataclass(unsafe_hash=True)
class Normal(ModelBase):
    r"""1D Gaussian with mixture weight.

    :math:`(weight, \mu, \sigma)(\phi1)`

    Parameters
    ----------
    n_layers : int, optional
        Number of hidden layers, by default 3.
    hidden_features : int, optional
        Number of hidden features, by default 50.
    sigma_upper_limit : float, optional keyword-only
        Upper limit on sigma, by default 0.3.
    fraction_upper_limit : float, optional keyword-only
        Upper limit on fraction, by default 0.45.s
    """

    _: KW_ONLY
    param_names: ParamNamesField = ParamNamesField(
        (WEIGHT_NAME, (..., ("mu", "sigma")))
    )
    param_bounds: ParamBoundsField[Array] = ParamBoundsField[Array](
        {  # reasonable guess for parameter bounds
            WEIGHT_NAME: SigmoidBounds(1e-10, 0.5),
            # ...: {"mu": SigmoidBounds(-5.0, 5.0), "sigma": SigmoidBounds(0.05, 1.5)},
        }
    )

    def __post_init__(self, array_namespace: ArrayNamespace[Array]) -> None:
        super().__post_init__(array_namespace=array_namespace)

        # Validate the coord_names
        if len(self.coord_names) != 1:
            msg = "Only one coordinate is supported, e.g ('phi2',)"
            raise ValueError(msg)

    def _net_init_default(self) -> NNModel:
        # Initialize the network
        # Note; would prefer nn.Parameter(xp.zeros((1, n_slopes)) + 1e-5)
        # as that has 1/2 as many params, but it's not callable.
        # TODO: ensure n_out == n_slopes
        # TODO! for jax need to bundle into 1 arg. Detect this!
        return self.xpnn.Sequential(
            self.xpnn.Linear(1, 36),
            self.xpnn.Tanh(),
            self.xpnn.Linear(36, 36),
            self.xpnn.Tanh(),
            self.xpnn.Linear(36, 3),
        )

    # ========================================================================
    # Statistics

    def ln_likelihood(
        self, mpars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Log-likelihood of the stream.

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
        data = self.data_scaler.transform(data, names=self.data_scaler.names)
        c = self.coord_names[0]
        lnlik = norm_logpdf(
            data[c], mpars[c, "mu"], xp.clip(mpars[c, "sigma"], min=1e-10), xp=self.xp
        )
        return xp.log(xp.clip(mpars[(WEIGHT_NAME,)], min=1e-10)) + lnlik
