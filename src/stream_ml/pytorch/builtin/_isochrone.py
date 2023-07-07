"""Built-in background models."""

from __future__ import annotations

__all__: list[str] = []

from dataclasses import KW_ONLY, dataclass, field
from functools import reduce
from typing import TYPE_CHECKING, Any, Final, Protocol

import torch as xp
from torch.distributions import MultivariateNormal

from stream_ml.core.utils.frozen_dict import FrozenDict, FrozenDictField
from stream_ml.core.utils.funcs import within_bounds

from stream_ml.pytorch import Data
from stream_ml.pytorch._base import ModelBase

if TYPE_CHECKING:
    from scipy.interpolate import CubicSpline

    from stream_ml.core.typing import BoundsT

    from stream_ml.pytorch.params import Params
    from stream_ml.pytorch.typing import Array, ArrayNamespace

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
        The names of the coordinates.
        This is not the same as the ``mag_names``.

    mag_names : tuple[str, ...], optional
        The names of the magnitudes.
    mag_err_names : tuple[str, ...], optional
        The names of the magnitude errors.

    gamma_edges : Array
        The edges of the gamma bins. Must be 1D.
    isochrone_spl : CubicSpline
        The isochrone spline for the one-to-one mapping of gamma to the
        magnitudes (``mag_names``).
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
    """

    _: KW_ONLY
    coord_names: tuple[str, ...] = ()  # optional

    # Photometric information
    mag_names: tuple[str, ...] = ("g", "r")
    mag_err_names: tuple[str, ...] = ("g_err", "r_err")
    mag_bounds: FrozenDictField[str, BoundsT] = FrozenDictField(FrozenDict())

    # Isochrone information
    gamma_edges: Array
    isochrone_spl: CubicSpline
    isochrone_err_spl: CubicSpline | None = None

    stream_mass_function: StreamMassFunction = field(
        default_factory=UniformStreamMassFunction
    )

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        super().__post_init__(*args, **kwargs)

        # Coordinate bounds are necessary
        if self.mag_bounds.keys() != set(self.mag_names):
            msg = (
                f"`mag_bounds` ({tuple(self.mag_bounds.keys())}) do not match "
                f"`mag_names` ({self.mag_names})."
            )
            raise ValueError(msg)

        # mag_err_names must be None or the same length as mag_names.
        # we can't check that the names are the same, because they aren't.
        kmen = self.mag_err_names
        if kmen is not None and len(kmen) != len(self.mag_names):
            msg = (
                f"`mag_err_names` ({kmen}) must be None or "
                f"the same length as `mag_names` ({self.mag_names})."
            )
            raise ValueError(msg)

        # check gamma_edges:
        if self.gamma_edges[0] != 0 or self.gamma_edges[-1] != 1:
            msg = "gamma_edges must start with 0 and end with 1"
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

    def _mags_in_bound(self, data: Data[Array], /) -> Array:
        """Elementwise log prior for coordinate bounds.

        Zero everywhere except where the data are outside the
        coordinate bounds, where it is -inf.
        """
        shape = data.array.shape[:1] + data.array.shape[2:]
        where = reduce(
            self.xp.logical_or,
            (~within_bounds(data[k], *v) for k, v in self.mag_bounds.items()),
            self.xp.zeros(shape, dtype=bool),
        )
        return self.xp.where(
            where, self.xp.full(shape, -self.xp.inf), self.xp.zeros(shape)
        )

    def ln_likelihood(
        self, mpars: Params[Array], /, data: Data[Array], **kwargs: Array
    ) -> Array:
        """Compute the log-likelihood.

        Parameters
        ----------
        mpars : Params
            The model parameters. Must contain the following parameters:

            - distmod, mu : [mag]
            - distmod, ln-sigma : [mag]

        data : Data[Array[(N,)]]
            The data. Must contain the fields in ``mag_names`` and ``mag_err_names``.
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
        mean = self._isochrone_locs + dm[:, None, None]

        # Compute what gamma is observable for each star
        # For non-observable stars, set the ln-weight to -inf
        # (N, F, I)
        mean_data = Data(xp.swapaxes(mean, 1, 2), names=self.mag_names)
        in_bounds = self._mags_in_bound(mean_data)  # (N, I)

        # log prior: the cluster mass function (N, I)
        ln_cmf = self.stream_mass_function(
            self._gamma_points, data[self.indep_coord_names], xp=self.xp
        )

        # Covariance: star (N, [I], F, F)
        cov_data = xp.diag_embed(data[self.mag_err_names].array ** 2)[:, None, :, :]
        # Covariance: isochrone ([N], I, F, F)
        # Covariance: distance modulus  (N, [I], F, F)
        cov_dm = xp.diag_embed(
            xp.ones((len(data), len(self.mag_names))) * dm_sigma[:, None] ** 2
        )[:, None, :, :]
        cov = cov_data + self._isochrone_cov + cov_dm

        # Log-likelihood of the multivariate normal
        mvn = MultivariateNormal(mean, covariance_matrix=cov)
        mdata = data[self.mag_names].array[:, None, ...]  # (N, [I], F)
        lnliks = mvn.log_prob(mdata)  # (N, I)

        # log PDF: the (log)-Reimannian sum over the isochrone (log)-pdfs:
        # sum_i(deltagamma_i PDF(gamma_i) * Pgamma)  -> translated to log_pdf
        return xp.logsumexp(
            self._ln_d_gamma + lnliks + ln_cmf + in_bounds, 1
        ) - xp.logsumexp(self._ln_d_gamma + ln_cmf + in_bounds, 1)
