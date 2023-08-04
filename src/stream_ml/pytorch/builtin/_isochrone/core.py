"""Built-in background models."""

from __future__ import annotations

__all__ = ["IsochroneMVNorm"]

from dataclasses import KW_ONLY, dataclass, field
from functools import reduce
from typing import TYPE_CHECKING, Any

import torch as xp

from stream_ml.core._core.field import NNField
from stream_ml.core.builtin._utils import WhereRequiredError
from stream_ml.core.utils.frozen_dict import FrozenDict, FrozenDictField
from stream_ml.core.utils.funcs import within_bounds

from stream_ml.pytorch import Data
from stream_ml.pytorch._base import ModelBase
from stream_ml.pytorch.builtin._isochrone.mf import (
    StreamMassFunction,
    UniformStreamMassFunction,
)

if TYPE_CHECKING:
    from scipy.interpolate import CubicSpline

    from stream_ml.core.typing import BoundsT

    from stream_ml.pytorch.params import Params
    from stream_ml.pytorch.typing import Array, NNModel


_log2pi = xp.log(xp.asarray(2 * xp.pi))


@dataclass(unsafe_hash=True)
class IsochroneMVNorm(ModelBase):
    """Isochrone Multi-Variate Normal.

    Parameters
    ----------
    net : Module, keyword-only
        The network to use. If not provided, a new one will be created. Must map
        1->3 input to output.

    params : ModelParametersField, optional
        The parameters.
    indep_coord_names : tuple[str, ...], optional
        The names of the independent coordinates.
    coord_names : tuple[str, ...], optional
        The names of the coordinates. This is not the same as the
        ``phot_names``.

    phot_names : tuple[str, ...], optional
        The names of the photometric coordinates: magnitudes and colors.
    phot_err_names : tuple[str, ...], optional
        The names of the photometric errors: magnitudes and colors.
    phot_apply_dm : tuple[bool, ...], optional
        Whether to apply the distance modulus to the photometric coordinate.
        Must be the same length as ``phot_names``.

    gamma_edges : Array
        The edges of the gamma bins. Must be 1D.
    isochrone_spl : CubicSpline
        The isochrone spline for the one-to-one mapping of gamma to the
        photometry (``phot_names``).
    isochrone_err_spl : CubicSpline
        The isochrone spline for the one-to-one mapping of gamma to the
        magnitude errors.

    stream_mass_function : `StreamMassFunction`, optional
        The cluster mass function. Must be parametrized by gamma [0, 1], the
        normalized mass over the range of the isochrone. Defaults to a uniform
        distribution. Returns the log-probability that stars of that mass
        (gamma) are in the population modeled by the isochrone.

    Notes
    -----
    Ln-likelihood required parameters:

    - distmod, mu : [mag]
    - distmod, ln-sigma : [mag]

    Examples
    --------
    An example of how to use this model:

    ::

        from stream_ml.pytorch.builtin import IsochroneMVNorm

        model = IsochroneMVNorm(
            net=...,  # (N,) -> ... data_scaler=...,
            indep_coord_names=("phi1",), # coordinates coord_names=(...),
            coord_bounds=(...), # photometry mag_names=("g",),
            mag_err_names=("g_err",), color_names=("g-r",),
            color_err_names=("g-r_err",), phot_bounds=(...),
    """

    net: NNField[NNModel, None] = NNField(default=None)

    _: KW_ONLY
    coord_names: tuple[str, ...] = ()  # optional

    # Photometric information
    phot_names: tuple[str, ...]
    phot_err_names: tuple[str, ...] | None = None
    phot_apply_dm: tuple[bool, ...] = ()
    phot_bounds: FrozenDictField[str, BoundsT] = FrozenDictField(FrozenDict())

    # Isochrone information
    gamma_edges: Array
    isochrone_spl: CubicSpline
    isochrone_err_spl: CubicSpline | None = None

    stream_mass_function: StreamMassFunction = field(
        default_factory=UniformStreamMassFunction
    )

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        super().__post_init__(*args, **kwargs)

        # Need phot_names
        if not self.phot_names:
            msg = "Must provide `phot_names`."
            raise ValueError(msg)
        elif not isinstance(self.phot_names, tuple):
            msg = "`phot_names` must be a tuple."  # type: ignore[unreachable]
            raise ValueError(msg)

        # And phot_apply_dm
        if len(self.phot_apply_dm) != len(self.phot_names):
            msg = (
                f"`phot_apply_dm` ({self.phot_apply_dm}) must be the same "
                f"length as `phot_names` ({self.phot_names})."
            )
            raise ValueError(msg)

        # phot_err_names must be None or the same length as phot_names.
        # we can't check that the names are the same, because they aren't.
        kpen = self.phot_err_names
        if kpen is not None and len(kpen) != len(self.phot_names):
            msg = (
                f"`phot_err_names` ({kpen}) must be None or "
                f"the same length as `phot_names` ({self.phot_names})."
            )
            raise ValueError(msg)

        # Coordinate bounds are necessary
        if self.phot_bounds.keys() != set(self.phot_names):
            msg = (
                f"`phot_bounds` ({tuple(self.phot_bounds.keys())}) do not match "
                f"`phot_names` ({self.phot_names})."
            )
            raise ValueError(msg)

        # Check gamma_edges:
        if (self.gamma_edges[0] != 0) or (self.gamma_edges[-1] != 1):
            msg = "gamma_edges must start with 0 and end with 1"
            raise ValueError(msg)

        # Check isochrone_spl:
        if self.isochrone_spl.c.shape[-1] != len(self.phot_names):
            msg = (
                f"`isochrone_spl` must have {len(self.phot_names)} "
                f"features, but has {self.isochrone_spl.c.shape[-1]}."
            )
            raise ValueError(msg)

        # Pairwise distance along gamma  # ([N], I)
        gamma_pdist = self.gamma_edges[1:] - self.gamma_edges[:-1]
        self._ln_d_gamma = self.xp.log(gamma_pdist[None, :])
        # Midpoint of gamma edges array
        self._gamma_points: Array = (self.gamma_edges[:-1] + self.gamma_edges[1:]) / 2
        # Points on the isochrone along gamma  # ([N], I, F)
        isochrone_locs = xp.asarray(
            self.isochrone_spl(self._gamma_points), dtype=self._gamma_points.dtype
        )
        self._isochrone_locs = isochrone_locs[None, :, :]
        # And errors  ([N], I, F, F)
        if self.isochrone_err_spl is None:
            self._isochrone_cov = self.xp.asarray([0])[None, None, None]
        else:
            isochrone_err = xp.asarray(self.isochrone_err_spl(self._gamma_points))
            self._isochrone_cov = xp.diag_embed(isochrone_err[None, :, :])

    @property
    def nI(self) -> int:  # noqa: N802
        """The number of isochrone points."""
        return len(self._gamma_points)

    @property
    def nF(self) -> int:  # noqa: N802
        """The number of photometric features."""
        return len(self.phot_names)

    def _phot_in_bound(self, data: Data[Array], /) -> Array:
        """Elementwise log prior for coordinate bounds.

        Zero everywhere except where the data are outside the
        coordinate bounds, where it is -inf.
        """
        shape = data.array.shape[:1] + data.array.shape[2:]
        where = reduce(
            self.xp.logical_or,
            (~within_bounds(data[k], *v) for k, v in self.phot_bounds.items()),
            self.xp.zeros(shape, dtype=bool),
        )
        return self.xp.where(
            where, self.xp.full(shape, -self.xp.inf), self.xp.zeros(shape)
        )

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
        r"""Compute the log-likelihood.

        Parameters
        ----------
        mpars : Params
            The model parameters. Must contain the following parameters:

            - distmod, mu : [mag]
            - distmod, ln-sigma : [mag]

        data : Data[Array[(N,)]]
            The data. Must contain the fields in ``phot_names`` and
            ``phot_err_names``.

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

        **kwargs: Array
            Not used.

        Returns
        -------
        Array[(N,)]
        """
        # 'where' is used to indicate which data points are available. If
        # 'where' is not provided, then all data points are assumed to be
        # available.
        where_: Array  # (N, F)
        if where is not None:
            where_ = where[self.phot_names].array
        elif self.require_where:
            raise WhereRequiredError
        else:
            where_ = self.xp.ones((len(data), self.nF), dtype=bool)

        if correlation_matrix is not None and correlation_det is None:
            msg = "Must provide `correlation_det`."
            raise ValueError(msg)

        # Mean : isochrone + distance modulus
        dm = mpars[("distmod", "mu")]  # (N,)
        mean = self.xp.zeros((len(dm), 1, 1)) + self._isochrone_locs  # (N, I, F)
        mean[:, :, self.phot_apply_dm] += dm[:, None, None]

        # Compute what gamma is observable for each star
        # For non-observable stars, set the ln-weight to -inf
        # (N, F, I)
        mean_data = Data(xp.swapaxes(mean, 1, 2), names=self.phot_names)
        in_bounds = self._phot_in_bound(mean_data)  # (N, I)

        # log prior: the cluster mass function (N, I)
        ln_cmf = self.stream_mass_function(
            self._gamma_points, data[self.indep_coord_names], xp=self.xp
        )

        # Covariance: star (N, [I], F, F)
        stds = data[self.phot_err_names].array
        std_data = (
            xp.diag_embed(stds)[:, None, :, :]
            if self.phot_err_names is not None
            else self.xp.zeros(1)
        )
        # Covariance data "The covariance matrix can be written as the rescaling
        # of a correlation matrix by the marginal variances:"
        # (https://en.wikipedia.org/wiki/Covariance_matrix#Correlation_matrix)
        cov_data = (
            std_data**2
            if correlation_matrix is None
            else std_data @ correlation_matrix[:, None, :, :] @ std_data
        )
        # Covariance: isochrone ([N], I, F, F)
        dm_sigma = self.xp.exp(mpars[("distmod", "ln-sigma")])  # (N,)
        # Covariance: distance modulus  (N, [I], F, F)
        cov_dm = xp.diag_embed(xp.ones((len(data), self.nF)) * dm_sigma[:, None] ** 2)[
            :, None, :, :
        ]
        # The covariance, setting non-observed dimensions to 0.
        # (N, I, F, F) positive definite.
        idx_cov = xp.diag_embed(where_.to(dtype=data.dtype))[:, None]  # (N, I, F, F)
        cov = idx_cov @ (cov_data + self._isochrone_cov + cov_dm) @ idx_cov
        # The determinant, dropping the dimensionality of non-observed
        # dimensions.
        logdet = xp.log(
            xp.linalg.det(cov + (xp.eye(self.nF)[None, None] - idx_cov))
        )  # (N, [I])

        # Dimensionality, dropping missing dimensions (N, [I])
        D = where_.sum(dim=-1)  # noqa: N806

        # Construct the data - mean (N, I, F), setting non-observed dimensions to 0.
        sel = where_[:, None, :].expand(-1, self.nI, -1)
        x = xp.zeros((len(data), self.nI, self.nF), dtype=data.dtype)
        x[sel] = (
            data[self.phot_names].array[:, None, :].expand(-1, self.nI, -1)[sel]
            - mean[sel]
        )

        lnliks = xp.zeros((len(data), len(self._gamma_points)))  # (N, I)
        lnliks = -0.5 * (  # (N, I, 1, 1) -> (N, I)
            D[:, None] * _log2pi
            + logdet
            + (
                x[:, :, None, :]  # (N, I, 1, F)
                @ xp.linalg.pinv(cov)  # (N, I, F, F)
                @ x[..., None]  # (N, I, F, 1)
            )[..., 0, 0]
        )
        # mvn = MultivariateNormal(mean, covariance_matrix=cov)
        # mdata = data[self.phot_names].array[:, None, ...]  # (N, [I], F)
        # lnliks = mvn.log_prob(mdata)  # (N, I)

        # log PDF: the (log)-Reimannian sum over the isochrone (log)-pdfs:
        # sum_i(deltagamma_i PDF(gamma_i) * Pgamma)  -> translated to log_pdf
        lnlik_unnormalized = xp.logsumexp(
            self._ln_d_gamma + lnliks + ln_cmf + in_bounds, 1
        )
        normalization = xp.nan_to_num(
            xp.logsumexp(self._ln_d_gamma + ln_cmf + in_bounds, 1),
        )
        return lnlik_unnormalized - normalization
