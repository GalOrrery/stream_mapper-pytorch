"""Built-in background models."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Any, Final

from scipy.interpolate import CubicSpline  # noqa: TCH002
import torch as xp
from torch.distributions import MultivariateNormal

from stream_ml.core.params.names import ParamNamesField
from stream_ml.core.setup_package import WEIGHT_NAME
from stream_ml.core.utils.funcs import pairwise_distance
from stream_ml.core.utils.sentinel import MISSING
from stream_ml.pytorch.base import ModelBase
from stream_ml.pytorch.nn import lin_tanh

__all__: list[str] = []

if TYPE_CHECKING:
    from torch.nn import Module

    from stream_ml.core.data import Data
    from stream_ml.core.params import Params
    from stream_ml.pytorch.typing import Array

dm_sigma_const: Final = 5 / xp.log(xp.asarray(10))
# Constant for first order error propagation of parallax -> distance modulus


@dataclass(unsafe_hash=True)
class IsochroneMVNorm(ModelBase):
    """Isochrone Multi-Variate Normal.

    Parameters
    ----------
    net : Module, keyword-only
        The network to use. If not provided, a new one will be created. Must map
        1->3 input to output.

    gamma_edges : Array
        The edges of the gamma bins. Must be 1D.
    isochrone_spl : CubicSpline
        The isochrone spline for the one-to-one mapping of gamma to the
        magnitudes (``mag_names``).
    isochrone_err_spl : CubicSpline
        The isochrone spline for the one-to-one mapping of gamma to the
        magnitude errors.

    param_names : ParamNamesField, optional
        The names of the parameters.
    indep_coord_names : tuple[str, ...], optional
        The names of the independent coordinates.
    coord_names : tuple[str, ...], optional
        The names of the coordinates.
    mag_names : tuple[str, ...], optional
        The names of the magnitudes.
    mag_err_names : tuple[str, ...], optional
        The names of the magnitude errors.

    Notes
    -----
    Ln-likelihood required parameters:

    - weight
    - distmod, mu : [mag]
    - distmod, sigma : [mag]
    """

    _: KW_ONLY
    gamma_edges: Array
    isochrone_spl: CubicSpline
    isochrone_err_spl: CubicSpline

    param_names: ParamNamesField = ParamNamesField(MISSING)
    indep_coord_names: tuple[str, ...] = ("phi1",)
    coord_names: tuple[str, ...] = ()
    mag_names: tuple[str, ...] = ("g", "r")
    mag_err_names: tuple[str, ...] = ("g_err", "r_err")

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        super().__post_init__(*args, **kwargs)
        # Pairwise distance along gamma  # ([N], I, 1)
        gamma_pdist = pairwise_distance(self.gamma_edges, axis=0, xp=self.xp)
        self._gamma_pdist = gamma_pdist[None, :, None]
        # Midpoint of gamma edges array
        self._gamma_points: Array = (
            self.gamma_edges[:-1] + self._gamma_pdist[0, :, 0] / 2
        )
        # Points on the isochrone along gamma  # ([N], I, F)
        isochrone_locs = xp.asarray(self.isochrone_spl(self._gamma_points))
        self._isochrone_locs = isochrone_locs[None, :, :]
        # And errors  ([N], I, F, F)
        isochrone_err = xp.asarray(self.isochrone_err_spl(self._gamma_points))
        self._isochrone_cov = xp.diag_embed(isochrone_err[None, :, :])

    def _net_init_default(self) -> Module:
        # return self.xpnn.Identity()
        return lin_tanh(n_in=1, n_hidden=10, n_layers=2, n_out=3)

    def ln_likelihood(
        self, mpars: Params[Array], data: Data[Array], **kwargs: Array
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
        dm = mpars[("distmod", "mu")].reshape((-1, 1))
        dm_sigma = mpars[("distmod", "sigma")].reshape((-1, 1))

        # Mean : isochrone + distance modulus
        # ([N], I, F) + (N, [I], [F]) = (N, I, F)
        mean = self._isochrone_locs + dm.reshape((-1, 1, 1))
        # Covariance: star (N, [I], F, F)
        cov_data = xp.diag_embed(data[:, self.mag_err_names, 0] ** 2)[:, None, :, :]
        # Covariance: isochrone ([N], I, F, F)
        # Covariance: distance modulus  (N, [I], F, F)
        cov_dm = xp.diag_embed(xp.ones((len(data), 2)) * dm_sigma**2)[:, None, :, :]
        cov = cov_data + self._isochrone_cov + cov_dm

        mvn = MultivariateNormal(mean, covariance_matrix=cov)
        mdata = data[:, self.mag_names, 0]  # (N, F)
        lnliks = mvn.log_prob(mdata[:, None, :])[..., None]  # (N, I)

        # log PDF: the (log)-Reimannian sum over the isochrone (log)-pdfs:
        ln_wgt = xp.log(xp.clip(mpars[(WEIGHT_NAME,)], min=xp.tensor(1e-10)))
        # sum_i(deltagamma_i PDF(gamma_i)) / sum_i(deltagamma_i)  -> translated
        # to log_pdfs
        return (
            ln_wgt
            + xp.logsumexp(lnliks + xp.log(self._gamma_pdist), 1)
            - xp.log(self._gamma_pdist.sum(1))  # should be log(1)
        )
