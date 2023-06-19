"""Built-in background models."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING, Any, Final, Protocol

import torch as xp
from torch.distributions import MultivariateNormal

from stream_ml.core.data import Data
from stream_ml.core.utils.funcs import pairwise_distance

from stream_ml.pytorch._base import ModelBase

__all__: list[str] = []

if TYPE_CHECKING:
    from scipy.interpolate import CubicSpline

    from stream_ml.core.params import Params
    from stream_ml.core.typing import ArrayNamespace

    from stream_ml.pytorch.typing import Array

dm_sigma_const: Final = 5 / xp.log(xp.asarray(10))
# Constant for first order error propagation of parallax -> distance modulus


# =============================================================================
# Cluster Mass Function


class ClusterMassFunction(Protocol):
    """Cluster Mass Function.

    Must be parametrized by gamma [0, 1], the normalized mass over the range of the
    isochrone.
    """

    def __call__(self, gamma: Array, *, xp: ArrayNamespace[Array]) -> Array:
        """Call."""
        ...


class UniformClusterMassFunction(ClusterMassFunction):
    def __call__(self, gamma: Array, *, xp: ArrayNamespace[Array]) -> Array:
        return xp.zeros_like(gamma)


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

    cluster_mass_function : `ClusterMassFunction`, optional
        The cluster mass function. Must be parametrized by gamma [0, 1], the
        normalized mass over the range of the isochrone. Defaults to a uniform
        distribution. Returns the log-probability that stars of that mass
        (gamma) are in the population modeled by the isochrone.

    Notes
    -----
    Ln-likelihood required parameters:

    - weight
    - distmod, mu : [mag]
    - distmod, sigma : [mag]
    """

    _: KW_ONLY
    # Photometric information
    mag_names: tuple[str, ...] = ("g", "r")
    mag_err_names: tuple[str, ...] = ("g_err", "r_err")

    gamma_edges: Array
    isochrone_spl: CubicSpline
    isochrone_err_spl: CubicSpline | None = None

    cluster_mass_function: ClusterMassFunction = field(
        default_factory=UniformClusterMassFunction
    )

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        super().__post_init__(*args, **kwargs)

        # Check mag_names is a subset of coord_names
        if not set(self.mag_names).issubset(self.coord_names):
            msg = (
                f"mag_names {self.mag_names!r} must be a subset of "
                f"coord_names {self.coord_names!r}"
            )
            raise ValueError(msg)
        # Pairwise distance along gamma  # ([N], I)
        gamma_pdist = pairwise_distance(self.gamma_edges, axis=0, xp=self.xp)
        self._ln_gamma_pdist = self.xp.log(gamma_pdist[None, :])
        # Midpoint of gamma edges array
        self._gamma_points: Array = self.gamma_edges[:-1] + gamma_pdist / 2
        # Points on the isochrone along gamma  # ([N], I, F)
        isochrone_locs = xp.asarray(self.isochrone_spl(self._gamma_points))
        self._isochrone_locs = isochrone_locs[None, :, :]
        # And errors  ([N], I, F, F)
        if self.isochrone_err_spl is None:
            self._isochrone_cov = self.xp.asarray([0])[None, None, None]
        else:
            isochrone_err = xp.asarray(self.isochrone_err_spl(self._gamma_points))
            self._isochrone_cov = xp.diag_embed(isochrone_err[None, :, :])

    def ln_likelihood(
        self, mpars: Params[Array], /, data: Data[Array], **kwargs: Array
    ) -> Array:
        """Compute the log-likelihood.

        Parameters
        ----------
        mpars : Params
            The model parameters. Must contain the following parameters:

            - weight
            - distmod, mu : [mag]
            - distmod, sigma : [mag]

        data : Data[Array[(N,)]]
            The data. Must contain the fields in ``mag_names`` and ``mag_err_names``.
        **kwargs: Array
            Not used.

        Returns
        -------
        Array[(N,)]
        """
        dm = mpars[("distmod", "mu")]  # (N,)
        dm_sigma = mpars[("distmod", "sigma")]  # (N,)

        # Mean : isochrone + distance modulus
        # ([N], I, F) + (N, [I], [F]) = (N, I, F)
        mean = self._isochrone_locs + dm[:, None, None]

        # Compute what gamma is observable for each star
        # For non-observable stars, set the ln-weight to -inf
        mean_data = Data(xp.swapaxes(mean, 1, 2), names=self.mag_names)
        in_bounds = self._ln_prior_coord_bnds(mean_data)

        # log prior: the cluster mass function ([N], I)
        ln_cmf = self.cluster_mass_function(self._gamma_points, xp=self.xp)[None, :]

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
        return xp.logsumexp(self._ln_gamma_pdist + lnliks + ln_cmf + in_bounds, 1)
