"""Multivariate Gaussian model."""

from __future__ import annotations

__all__: tuple[str, ...] = ()

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch as xp

from stream_mapper.core.builtin import WhereRequiredError

from stream_mapper.pytorch._base import ModelBase

if TYPE_CHECKING:
    from stream_mapper.core import Data, Params

    from stream_mapper.pytorch.typing import Array


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
        where: Data[Array] | None = None,
        correlation_matrix: Array | None = None,
        correlation_det: Array | None = None,
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

        where : Data[Array[(N,), bool]] | None, optional keyword-only
            Where to evaluate the log-likelihood. If not provided, then the
            log-likelihood is evaluated at all data points. ``where`` must
            contain the fields in ``phot_names``. Each field must be a boolean
            array of the same length as `data`. `True` indicates that the data
            point is available, and `False` indicates that the data point is not
            available.

        correlation_matrix : Array[(N,F,F)] | None, optional keyword-only
            The correlation matrix. If not provided, then the covariance matrix
            is assumed to be diagonal. The covariance matrix is computed as:

            .. math::

                \rm{cov}(X) =       \rm{diag}(\vec{\sigma})
                              \cdot \rm{corr} \cdot \rm{diag}(\vec{\sigma})
        correlation_det: Array[(N,)] | None, optional keyword-only
            The determinant of the correlation matrix. If not provided, then
            the determinant is only the product of the diagonal elements of the
            covariance matrix.

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
            where_ = self.xp.ones((len(data), self.nF), dtype=bool)

        if correlation_matrix is not None and correlation_det is None:
            msg = "Must provide `correlation_det`."
            raise ValueError(msg)

        # Covariance data "The covariance matrix can be written as the rescaling
        # of a correlation matrix by the marginal variances:"
        # (https://en.wikipedia.org/wiki/Covariance_matrix#Correlation_matrix)
        std_data = (
            xp.diag_embed(data[self.coord_err_names].array)
            if self.coord_err_names is not None
            else self.xp.zeros(1)
        )
        cov_data = (
            std_data**2
            if correlation_matrix is None
            else std_data @ correlation_matrix[:, :, :] @ std_data
        )
        # Covariance model (N, F, F)
        lnsigma = self._stack_param(mpars, "ln-sigma", self.coord_names)
        cov_model = xp.diag_embed(self.xp.exp(2 * lnsigma))
        # The covariance, setting non-observed dimensions to 0. (N, F, F)
        # positive definite.
        idx_cov = xp.diag_embed(where_.to(dtype=data.dtype))
        cov = idx_cov @ (cov_data + cov_model) @ idx_cov
        # The determinant, dropping the dimensionality of non-observed
        # dimensions.
        logdet = xp.log(
            xp.linalg.det(cov + (xp.eye(self.nF)[None, None] - idx_cov))
        )  # (N, [I])

        # Dimensionality, dropping missing dimensions (N, [I])
        D = where_.sum(dim=-1)  # noqa: N806

        # Construct the data - mean (N, I, F), setting non-observed dimensions to 0.
        mu = self._stack_param(mpars, "mu", self.coord_names)
        sel = where_[:, None, :].expand(-1, self.nI, -1)
        x = xp.zeros((len(data), self.nI, self.nF), dtype=data.dtype)
        x[sel] = (
            data[self.coord_names].array[:, None, :].expand(-1, self.nI, -1)[sel]
            - mu[sel]
        )

        return -0.5 * (  # (N, I, 1, 1) -> (N, I)
            D * _log2pi
            + logdet
            + (
                x[:, None, :]  # (N, 1, F)
                @ xp.linalg.pinv(cov)  # (N,  F, F)
                @ x[..., None]  # (N, F, 1)
            )[..., 0, 0]
        )
