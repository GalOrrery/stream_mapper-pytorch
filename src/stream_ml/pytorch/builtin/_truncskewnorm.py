"""Gaussian stream model."""

from __future__ import annotations

__all__: list[str] = []

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

import torch as xp

from stream_ml.core.builtin._stats.trunc_norm import logpdf as truncnorm_logpdf
from stream_ml.core.builtin._stats.trunc_skewnorm import logpdf as truncskewnorm_logpdf
from stream_ml.core.builtin._utils import WhereRequiredError

from stream_ml.pytorch.builtin._skewnorm import SkewNormal

if TYPE_CHECKING:
    from stream_ml.pytorch import Data
    from stream_ml.pytorch.params import Params
    from stream_ml.pytorch.typing import Array


@dataclass(unsafe_hash=True)
class TruncatedSkewNormal(SkewNormal):
    r"""Truncated Skew-Normal."""

    _: KW_ONLY
    require_where: bool = False

    def ln_likelihood(
        self,
        mpars: Params[Array],
        /,
        data: Data[Array],
        *,
        where: Data[Array] | None = None,
        **kwargs: Array,
    ) -> Array:
        """Log-likelihood of the distribution.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array]
            Data (phi1, phi2).

        where : Data[Array], optional keyword-only
            Where to evaluate the log-likelihood. If not provided, then the
            log-likelihood is evaluated at all data points.
        **kwargs : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        # 'where' is used to indicate which data points are available. If
        # 'where' is not provided, then all data points are assumed to be
        # available.
        if where is not None:
            idx = where[tuple(self.coord_bounds.keys())].array
        elif self.require_where:
            raise WhereRequiredError
        else:
            idx = self.xp.ones(
                (len(data), len(self.coord_names)), dtype=bool, require_grad=False
            )
            # This has shape (N,F) so will broadcast correctly.

        cns, cens = self.coord_names, self.coord_err_names
        x = data[cns].array  # (N, F)
        _0 = self.xp.zeros_like(x)[None, ...]  # (1, N, F)

        a, b = _0 + self.xp.asarray([self.coord_bounds[k] for k in cns]).T[:, None, :]
        mu = self.xp.stack(tuple(mpars[(k, "mu")] for k in cns), 1)[idx]
        ln_s = self.xp.stack(tuple(mpars[(k, "ln-sigma")] for k in cns), 1)[idx]
        skew = self.xp.stack(tuple(mpars[(k, "skew")] for k in cns), 1)[idx]
        if cens is not None:
            # it's fine if sigma_o is 0
            # TODO: I suspect there are better ways to write this
            sigma_o = data[cens].array[idx]
            sigma = self.xp.exp(ln_s)
            skew = (
                skew * sigma / self.xp.sqrt(sigma**2 + (1 + skew**2) * sigma_o**2)
            )
            ln_s = self.xp.log(sigma**2 + sigma_o**2) / 2

        # Find where -inf
        with xp.no_grad():
            _lpdf = truncskewnorm_logpdf(
                x[idx], loc=mu, ln_sigma=ln_s, skew=skew, a=a[idx], b=b[idx], xp=self.xp
            )
            fnt = xp.isfinite(_lpdf)  # apply to X[idx] only

        # Compute SN where it's finite, to avoid numerical issues with the gradient
        sn_lnpdf = self.xp.full_like(x[idx], 0)
        sn_lnpdf[fnt] = truncskewnorm_logpdf(
            x[idx][fnt],
            loc=mu[fnt],
            ln_sigma=ln_s[fnt],
            skew=skew[fnt],
            a=a[idx][fnt],
            b=b[idx][fnt],
            xp=self.xp,
        )

        # Compute normal where SN is infinite.
        # Subtract 100 b/c that's where the SN logpdf drops to -inf
        n_lnpdf = (
            truncnorm_logpdf(
                x[idx], loc=mu, ln_sigma=ln_s, a=a[idx], b=b[idx], xp=self.xp
            )
            - 100
        )

        idxlnliks = xp.where(fnt, sn_lnpdf, n_lnpdf)

        lnliks = self.xp.full_like(x, 0)  # missing data is ignored
        lnliks[idx] = idxlnliks

        return lnliks.sum(1)
