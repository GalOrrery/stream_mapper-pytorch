"""Multivariate Gaussian model."""

from __future__ import annotations

__all__: list[str] = []

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

import torch as xp
from torch.distributions import MultivariateNormal as TorchMultivariateNormal

from stream_ml.pytorch._base import ModelBase

if TYPE_CHECKING:
    from stream_ml.pytorch import Data
    from stream_ml.pytorch.params import Params
    from stream_ml.pytorch.typing import Array


_log2pi = xp.log(xp.asarray(2 * xp.pi))


@dataclass(unsafe_hash=True)
class MultivariateNormal(ModelBase):
    """Multivariate-Normal Model."""

    # ========================================================================
    # Statistics

    def ln_likelihood(
        self,
        mpars: Params[Array],
        /,
        data: Data[Array],
        *,
        correlation_matrix: Array | None = None,
        **kwargs: Array,
    ) -> Array:
        r"""Log-likelihood of the distribution.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array]
            Data (phi1, phi2, ...).

        correlation_matrix : Array[(N,F,F)], optional keyword-only
            The correlation matrix. If not provided, then the covariance matrix is
            assumed to be diagonal.
            The covariance matrix is computed as:

            .. math::

                \rm{cov}(X) =       \rm{diag}(\vec{\sigma})
                              \cdot \rm{corr}
                              \cdot \rm{diag}(\vec{\sigma})

        **kwargs : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        marginals = xp.diag_embed(
            self.xp.exp(self._stack_param(mpars, "ln-sigma", self.coord_names))
        )
        cov = (
            marginals @ marginals
            if correlation_matrix is None
            else marginals @ correlation_matrix @ marginals
        )

        return TorchMultivariateNormal(
            self._stack_param(mpars, "mu", self.coord_names),
            covariance_matrix=cov,
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
        x = data[self.coord_names].array
        mu = self._stack_param(mpars, "mu", self.coord_names)
        sigma = self.xp.exp(self._stack_param(mpars, "ln-sigma", self.coord_names))

        idx: Array
        if mask is not None:
            idx = mask[tuple(self.coord_bounds.keys())].array
        elif self.require_mask:
            msg = "mask is required"
            raise ValueError(msg)
        else:
            idx = xp.ones_like(x, dtype=xp.int)
            # shape (1, F) so that it can broadcast with (N, F)

        D = idx.sum(dim=1)  # Dimensionality (N,)  # noqa: N806
        dmm = idx * (x - mu)  # Data - model (N, F)

        # Covariance related
        cov = idx * sigma**2  # (N, F) positive definite
        det = (cov * idx + (1 - idx)).prod(dim=1)  # (N,)

        return -0.5 * (
            D * _log2pi
            + xp.log(det)
            + (
                dmm[:, None, :]  # (N, 1, F)
                @ xp.linalg.pinv(xp.diag_embed(cov))  # (N, F, F)
                @ dmm[..., None]  # (N, F, 1)
            ).flatten()  # (N, 1, 1) -> (N,)
        )  # (N,)
