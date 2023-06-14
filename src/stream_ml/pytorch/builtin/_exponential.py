"""Built-in background models."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

from stream_ml.pytorch._base import ModelBase

__all__: list[str] = []


if TYPE_CHECKING:
    from stream_ml.core.data import Data
    from stream_ml.core.params import Params

    from stream_ml.pytorch.typing import Array


@dataclass(unsafe_hash=True)
class Exponential(ModelBase):
    r"""Tilted separately in each dimension.

    In each dimension the background is an exponential distribution between
    points ``a`` and ``b``. The rate parameter is ``m``.

    The non-zero portion of the PDF, where :math:`a < x < b` is

    .. math::

        f(x) = \frac{m * e^{-m * (x -a)}}{1 - e^{-m * (b - a)}}

    However, we use the order-3 Taylor expansion of the exponential function
    around m=0, to avoid the m=0 indeterminancy.

    .. math::

        f(x) =   \frac{1}{b-a}
               + m * (0.5 - \frac{x-a}{b-a})
               + \frac{m^2}{2} * (\frac{b-a}{6} - (x-a) + \frac{(x-a)^2}{b-a})
               + \frac{m^3}{12(b-a)} (2(x-a)-(b-a))(x-a)(b-x)
    """

    _: KW_ONLY
    require_mask: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()

        # Pre-compute the associated constant factors
        _b, _bma = [], []
        for k, (a, b) in self.coord_bounds.items():
            if k not in self.params.keys():
                continue
            _b.append(b)
            _bma.append(b - a)

        self._b = self.xp.asarray(_b)[None, :]
        self._bma = self.xp.asarray(_bma)[None, :]

    # ========================================================================
    # Statistics

    def ln_likelihood(
        self,
        mpars: Params[Array],
        /,
        data: Data[Array],
        *,
        mask: Data[Array] | None = None,
        **kwargs: Array,
    ) -> Array:
        """Log-likelihood of the background.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array]
            Labelled data.

        mask : (N, 1) Data[Array[bool]], keyword-only
            Data availability. True if data is available, False if not.
            Should have the same keys as `data`.
        **kwargs : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        # The mask is used to indicate which data points are available. If the
        # mask is not provided, then all data points are assumed to be
        # available.
        if mask is not None:
            indicator = mask[tuple(self.coord_bounds.keys())].array[..., 0]
        elif self.require_mask:
            msg = "mask is required"
            raise ValueError(msg)
        else:
            indicator = self.xp.ones((len(data), 1), dtype=int)
            # This has shape (N, 1) so will broadcast correctly.

        # Data is x - a
        d_arr = self._b - data[self.coord_names].array[..., 0]
        # Get the slope from `mpars` we check param names to see if the
        # slope is a parameter. If it is not, then we assume it is 0.
        # When the slope is 0, the log-likelihood reduces to a Uniform.
        ms = self.xp.hstack(
            tuple(
                mpars[(k, "slope")]
                if (k, "slope") in self.params.flatskeys()
                else self.xp.zeros((len(d_arr), 1))
                for k in self.coord_names
            )
        )
        n0 = ms != 0
        bma = self.xp.zeros_like(d_arr) + self._bma

        # log-likelihood
        lnliks = self.xp.zeros_like(d_arr)
        lnliks[~n0] = -self.xp.log(bma[~n0])  # Uniform
        lnliks[n0] = (
            self.xp.log(ms[n0] / self.xp.expm1(ms[n0] * bma[n0])) + ms[n0] * d_arr[n0]
        )

        return (indicator * lnliks).sum(1, keepdim=True)
