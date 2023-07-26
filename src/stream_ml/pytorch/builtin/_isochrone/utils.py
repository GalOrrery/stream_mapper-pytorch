"""Built-in background models."""

from __future__ import annotations

__all__: list[str] = []

from dataclasses import KW_ONLY, dataclass
from math import log
from typing import TYPE_CHECKING, Final

import torch as xp

from stream_ml.pytorch.params import set_param

if TYPE_CHECKING:
    from stream_ml.pytorch.params import Params
    from stream_ml.pytorch.typing import Array, ArrayNamespace


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
