"""Built-in background models."""

from __future__ import annotations

__all__: list[str] = []

from dataclasses import KW_ONLY, dataclass, field
from functools import reduce
from math import log
from typing import TYPE_CHECKING, Any, Final, Protocol

import torch as xp
from torch.distributions import MultivariateNormal

from stream_ml.core._core.field import NNField
from stream_ml.core.utils.frozen_dict import FrozenDict, FrozenDictField
from stream_ml.core.utils.funcs import within_bounds

from stream_ml.pytorch import Data
from stream_ml.pytorch._base import ModelBase
from stream_ml.pytorch.params import set_param

if TYPE_CHECKING:
    from scipy.interpolate import CubicSpline

    from stream_ml.core.typing import BoundsT

    from stream_ml.pytorch.params import Params
    from stream_ml.pytorch.typing import Array, ArrayNamespace, NNModel

dm_sigma_const: Final = 5 / xp.log(xp.asarray(10))
# Constant for first order error propagation of parallax -> distance modulus


# =============================================================================
# Cluster Mass Function


class StreamMassFunction(Protocol):
    """Stream Mass Function.

    Must be parametrized by gamma [0, 1], the normalized mass over the range of the
    isochrone.

    Returns the log-probability that stars of that mass (gamma) are in the
    population modeled by the isochrone.
    """

    def __call__(
        self, gamma: Array, x: Data[Array], *, xp: ArrayNamespace[Array]
    ) -> Array:
        r"""Log-probability of stars at position 'x' having mass 'gamma'.

        Parameters
        ----------
        gamma : Array[(F,))]
            The mass of the stars, normalized to [0, 1] over the range of the
            isochrone.
        x : Data[Array[(N,)]]
            The independent data. Normally this is :math:`\phi_1`.

        xp : ArrayNamespace[Array], keyword-only
            The array namespace.

        Returns
        -------
        Array[(N, F)]
        """
        ...


@dataclass(frozen=True)
class UniformStreamMassFunction(StreamMassFunction):
    def __call__(
        self, gamma: Array, x: Data[Array], *, xp: ArrayNamespace[Array]
    ) -> Array:
        return xp.zeros((len(x), len(gamma)))


@dataclass(frozen=True)
class HardCutoffMassFunction(StreamMassFunction):
    """Hard Cutoff IMF."""

    lower: float = 0
    upper: float = 1

    def __call__(
        self, gamma: Array, x: Data[Array], *, xp: ArrayNamespace[Array]
    ) -> Array:
        out = xp.full((len(x), len(gamma)), -xp.inf)
        out[:, (gamma >= self.lower) & (gamma <= self.upper)] = 0
        return out


@dataclass(frozen=True)
class StepwiseMassFunction(StreamMassFunction):
    """Hard Cutoff IMF."""

    boundaries: tuple[float, ...]  # (B + 1,)
    log_probs: tuple[float, ...]  # (B,)

    def __call__(
        self, gamma: Array, x: Data[Array], *, xp: ArrayNamespace[Array]
    ) -> Array:
        out = xp.full((len(x), len(gamma)), -xp.inf)
        for lower, upper, lnp in zip(
            self.boundaries[:-1], self.boundaries[1:], self.log_probs, strict=True
        ):
            out[:, (gamma >= lower) & (gamma < upper)] = lnp
        return out


# =============================================================================
# Isochrone Model


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

        # And phot_apply_dm
        if len(self.phot_apply_dm) != len(self.phot_names):
            msg = (
                f"`phot_apply_dm` ({self.phot_apply_dm}) must be the same "
                f"length as `phot_names` ({self.phot_names})."
            )
            raise ValueError(msg)

        # phot_err_names must be None or the same length as phot_names.
        # we can't check that the names are the same, because they aren't.
        kmen = self.phot_err_names
        if kmen is not None and len(kmen) != len(self.phot_names):
            msg = (
                f"`phot_err_names` ({kmen}) must be None or "
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
        isochrone_locs = xp.asarray(self.isochrone_spl(self._gamma_points))
        self._isochrone_locs = isochrone_locs[None, :, :]
        # And errors  ([N], I, F, F)
        if self.isochrone_err_spl is None:
            self._isochrone_cov = self.xp.asarray([0])[None, None, None]
        else:
            isochrone_err = xp.asarray(self.isochrone_err_spl(self._gamma_points))
            self._isochrone_cov = xp.diag_embed(isochrone_err[None, :, :])

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
        correlation_matrix: Array | None = None,
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
            The data. Must contain the fields in ``phot_names`` and ``phot_err_names``.

        correlation_matrix : Array[(N,F,F)], optional keyword-only
            The correlation matrix. If not provided, then the covariance matrix is
            assumed to be diagonal.
            The covariance matrix is computed as:

            .. math::

                \rm{cov}(X) =       \rm{diag}(\vec{\sigma})
                              \cdot \rm{corr}
                              \cdot \rm{diag}(\vec{\sigma})
        **kwargs: Array
            Not used.

        Returns
        -------
        Array[(N,)]
        """
        dm = mpars[("distmod", "mu")]  # (N,)
        dm_sigma = self.xp.exp(mpars[("distmod", "ln-sigma")])  # (N,)

        # Mean : isochrone + distance modulus
        # ([N], I, F) + (N, [I], [F]) = (N, I, F)
        mean = self.xp.zeros((len(dm), 1, 1)) + self._isochrone_locs
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
        vars_data = (
            xp.diag_embed(data[self.phot_err_names].array)[:, None, :, :]
            if self.phot_err_names is not None
            else self.xp.zeros(1)
        )
        cov_data = (
            vars_data @ vars_data
            if correlation_matrix is None
            else vars_data @ correlation_matrix[:, None, :, :] @ vars_data
        )
        # Covariance: isochrone ([N], I, F, F)
        # Covariance: distance modulus  (N, [I], F, F)
        cov_dm = xp.diag_embed(
            xp.ones((len(data), len(self.phot_names))) * dm_sigma[:, None] ** 2
        )[:, None, :, :]
        cov = cov_data + self._isochrone_cov + cov_dm

        # Log-likelihood of the multivariate normal
        mvn = MultivariateNormal(mean, covariance_matrix=cov)
        mdata = data[self.phot_names].array[:, None, ...]  # (N, [I], F)
        lnliks = mvn.log_prob(mdata)  # (N, I)

        # log PDF: the (log)-Reimannian sum over the isochrone (log)-pdfs:
        # sum_i(deltagamma_i PDF(gamma_i) * Pgamma)  -> translated to log_pdf
        lnlik_unnormalized = xp.logsumexp(
            self._ln_d_gamma + lnliks + ln_cmf + in_bounds, 1
        )
        normalization = xp.nan_to_num(
            xp.logsumexp(self._ln_d_gamma + ln_cmf + in_bounds, 1),
        )
        return lnlik_unnormalized - normalization


# =============================================================================

_five_over_log10: Final = 5 / log(10)


@dataclass(frozen=True)
class Parallax2DistMod:
    astrometric_coord: str
    photometric_coord: str

    _: KW_ONLY
    neg_clip_mu: float = 1e-30
    xp: ArrayNamespace[Array] = xp

    def __call__(self, pars: Params[Array], /) -> Params[Array]:
        # Convert parallax (mas) to distance modulus
        # .. math::
        #       distmod = 5 log10(d [pc]) - 5 = -5 log10(plx [arcsec]) - 5
        #               = -5 log10(plx [mas] / 1e3) - 5
        #               = 10 - 5 log10(plx [mas])
        # dm = 10 - 5 * xp.log10(pars["photometric.parallax"]["mu"].reshape((-1, 1)))
        dm = 10 - 5 * self.xp.log10(
            self.xp.clip(pars[self.astrometric_coord]["mu"], self.neg_clip_mu)
        )
        ln_dm_sigma = self.xp.log(
            _five_over_log10
            * self.xp.exp(pars[self.astrometric_coord]["ln-sigma"])
            * dm
        )

        # Set the distance modulus
        set_param(pars, (self.photometric_coord, "mu"), dm)
        set_param(pars, (self.photometric_coord, "ln-sigma"), ln_dm_sigma)

        return pars
