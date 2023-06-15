"""Multivariate Gaussian model."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

import torch as xp
from torch.distributions import MultivariateNormal as TorchMultivariateNormal

from stream_ml.pytorch._base import ModelBase

if TYPE_CHECKING:
    from stream_ml.core.data import Data
    from stream_ml.core.params import Params

    from stream_ml.pytorch.typing import Array

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

    # ========================================================================
    # Statistics

    def ln_likelihood(
        self, mpars: Params[Array], /, data: Data[Array], **kwargs: Array
    ) -> Array:
        """Log-likelihood of the distribution.

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
        return TorchMultivariateNormal(
            self.xp.stack([mpars[c, "mu"] for c in self.coord_names], 1),
            covariance_matrix=xp.diag_embed(
                self.xp.stack([mpars[c, "sigma"] for c in self.coord_names], 1) ** 2
            ),
        ).log_prob(data[self.coord_names].array)


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
        /,
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
        # Normal
        datav = data[self.coord_names].array
        mu = self.xp.stack([mpars[c, "mu"] for c in self.coord_names], 1)
        sigma = self.xp.stack([mpars[c, "sigma"] for c in self.coord_names], 1)

        indicator: Array
        if mask is not None:
            indicator = mask[tuple(self.coord_bounds.keys())].array
        elif self.require_mask:
            msg = "mask is required"
            raise ValueError(msg)
        else:
            indicator = xp.ones_like(datav, dtype=xp.int)
            # shape (1, F) so that it can broadcast with (N, F)

        # misc
        dimensionality = indicator.sum(dim=1)  # (N,)

        # Data - model
        dmm = indicator * (datav - mu)  # (N, F)

        # Covariance related
        cov = indicator * sigma**2  # (N, F) positive definite
        det = (cov + (1 - indicator)).prod(dim=1)  # (N,)

        return -0.5 * (
            dimensionality * _log2pi  # dim of data
            + xp.log(det)
            + (
                dmm[:, None, :]  # (N, 1, F)
                @ xp.linalg.pinv(xp.diag_embed(cov))  # (N, F, F)
                @ dmm[..., None]  # (N, F, 1)
            ).flatten()  # (N, 1, 1) -> (N,)
        )  # (N,)
