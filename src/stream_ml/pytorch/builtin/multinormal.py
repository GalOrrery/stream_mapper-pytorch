"""Multivariate Gaussian model."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

import torch as xp
from torch.distributions import MultivariateNormal as TorchMultivariateNormal

from stream_ml.core.params.bounds import ParamBoundsField
from stream_ml.core.params.names import ParamNamesField
from stream_ml.core.setup_package import WEIGHT_NAME
from stream_ml.core.utils.scale.utils import rescale
from stream_ml.pytorch.base import ModelBase
from stream_ml.pytorch.prior.bounds import SigmoidBounds
from stream_ml.pytorch.typing import Array, NNModel

if TYPE_CHECKING:
    from stream_ml.core.data import Data
    from stream_ml.core.params import Params

__all__: list[str] = []


_log2pi = xp.log(xp.asarray(2 * xp.pi))


@dataclass(unsafe_hash=True)
class MultivariateNormal(ModelBase):
    """Stream Model.

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

    def _net_init_default(self) -> NNModel:
        # Initialize the network
        # Note; would prefer nn.Parameter(xp.zeros((1, n_slopes)) + 1e-5)
        # as that has 1/2 as many params, but it's not callable.
        # TODO: ensure n_out == n_slopes
        # TODO! for jax need to bundle into 1 arg. Detect this!
        nout = 1 + 2 * len(self.coord_names)  # weight + (mu + sigma) * per coord

        return self.xpnn.Sequential(
            self.xpnn.Linear(1, 36),
            self.xpnn.Tanh(),
            self.xpnn.Linear(36, 36),
            self.xpnn.Tanh(),
            self.xpnn.Linear(36, nout),
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
            Data (phi1, phi2, ...).
        **kwargs : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        data = self.data_scaler.transform(data, names=self.data_scaler.names)
        mpars = rescale(self, mpars)

        ln_wgt = xp.log(xp.clip(mpars[(WEIGHT_NAME,)], 1e-10))
        datav = data[:, self.coord_names, 0]

        lik = TorchMultivariateNormal(
            xp.hstack([mpars[c, "mu"] for c in self.coord_names]),
            covariance_matrix=xp.diag_embed(
                xp.hstack([mpars[c, "sigma"] for c in self.coord_names]) ** 2
            ),
        ).log_prob(datav)

        return ln_wgt + lik[:, None]


##############################################################################


@dataclass(unsafe_hash=True)
class MultivariateMissingNormal(MultivariateNormal):  # (MultivariateNormal)
    """Multivariate Normal with missing data.

    .. note::

        Currently this requires a diagonal covariance matrix.
    """

    _: KW_ONLY
    require_mask: bool = True

    def ln_likelihood(
        self,
        mpars: Params[Array],
        data: Data[Array],
        *,
        mask: Data[Array] | None = None,
        **kwargs: Array,
    ) -> Array:
        """Negative log-likelihood.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array]
            Labelled data.
        mask : Data[Array[bool]] | None, optional
            Data availability. `True` if data is available, `False` if not.
            Should have the same keys as `data`.
        **kwargs : Array
            Additional arguments.
        """
        data = self.data_scaler.transform(data, names=self.data_scaler.names)
        mpars = rescale(self, mpars)

        # Mixture
        ln_wgt = xp.log(xp.clip(mpars[(WEIGHT_NAME,)], min=1e-10))

        # Normal
        datav = data[:, self.coord_names, 0]
        mu = xp.hstack([mpars[c, "mu"] for c in self.coord_names])
        sigma = xp.hstack([mpars[c, "sigma"] for c in self.coord_names])

        if mask is not None:
            indicator = mask[:, tuple(self.coord_bounds.keys()), 0]
        elif self.require_mask:
            msg = "mask is required"
            raise ValueError(msg)
        else:
            indicator = xp.ones_like(datav, dtype=xp.int)
            # shape (1, F) so that it can broadcast with (N, F)

        # misc
        dimensionality = indicator.sum(dim=1, keepdim=True)  # (N, 1)

        # Data - model
        dmm = indicator * (datav - mu)  # (N, 4)

        # Covariance related
        cov = indicator * sigma**2  # (N, 4) positive definite  # TODO: add eps
        det = (cov + (1 - indicator)).prod(dim=1, keepdims=True)  # (N, 1)

        return ln_wgt - 0.5 * (
            dimensionality * _log2pi  # dim of data
            + xp.log(det)
            + (  # TODO: speed up
                dmm[:, None, :]  # (N, 1, 4)
                @ xp.linalg.pinv(xp.diag_embed(cov))  # (N, 4, 4)
                @ dmm[:, :, None]  # (N, 4, 1)
            )[:, :, 0]
        )  # (N, 1)
