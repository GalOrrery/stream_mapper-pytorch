"""Built-in background models."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

from stream_ml.core.params.bounds import ParamBoundsField
from stream_ml.core.params.names import ParamNamesField
from stream_ml.core.prior.bounds import ClippedBounds
from stream_ml.core.setup_package import WEIGHT_NAME
from stream_ml.core.utils.scale.utils import rescale
from stream_ml.pytorch.base import ModelBase
from stream_ml.pytorch.prior.bounds import SigmoidBounds
from stream_ml.pytorch.typing import Array, NNModel

__all__: list[str] = []


if TYPE_CHECKING:
    from stream_ml.core.data import Data
    from stream_ml.core.params import Params
    from stream_ml.core.typing import ArrayNamespace


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
    param_names: ParamNamesField = ParamNamesField(
        (WEIGHT_NAME, (..., ("slope",))), requires_all_coordinates=False
    )
    param_bounds: ParamBoundsField[Array] = ParamBoundsField[Array](
        {
            WEIGHT_NAME: ClippedBounds(1e-10, 1.0, param_name=(WEIGHT_NAME,)),
            ...: {"slope": SigmoidBounds(-1.0, 1.0)},  # param_name is filled in later
        }
    )
    require_mask: bool = False

    def __post_init__(self, array_namespace: ArrayNamespace[Array]) -> None:
        super().__post_init__(array_namespace=array_namespace)

        # Pre-compute the associated constant factors
        _a, _bma = [], []
        for k, (a, b) in self.coord_bounds.items():
            if k not in self.param_names.top_level:
                continue
            _a.append(a)
            _bma.append(b - a)

        self._a = self.xp.asarray(_a)
        self._bma = self.xp.asarray(_bma)

    def _net_init_default(self) -> NNModel:
        # Initialize the network
        # Note; would prefer nn.Parameter(xp.zeros((1, n_slopes)) + 1e-5)
        # as that has 1/2 as many params, but it's not callable.
        # TODO: ensure n_out == n_slopes
        # TODO! for jax need to bundle into 1 arg. Detect this!
        return self.xpnn.Sequential(
            self.xpnn.Linear(1, len(self.param_names) - 1), self.xpnn.Sigmoid()
        )

    # ========================================================================
    # Statistics

    def ln_likelihood(
        self,
        mpars: Params[Array],
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
        ln_wgt = self.xp.log(self.xp.clip(mpars[(WEIGHT_NAME,)], 1e-10))

        # The mask is used to indicate which data points are available. If the
        # mask is not provided, then all data points are assumed to be
        # available.
        if mask is not None:
            indicator = mask[:, tuple(self.coord_bounds.keys()), 0]
        elif self.require_mask:
            msg = "mask is required"
            raise ValueError(msg)
        else:
            indicator = self.xp.ones_like(ln_wgt, dtype=self.xp.int)
            # This has shape (N, 1) so will broadcast correctly.

        # Data is x - a
        d_arr = data[:, self.coord_names, 0] - self._a
        # Get the slope from `mpars` we check param_names to see if the
        # slope is a parameter. If it is not, then we assume it is 0.
        # When the slope is 0, the log-likelihood reduces to a Uniform.
        ms = self.xp.hstack(
            tuple(
                mpars[(k, "slope")]
                if (k, "slope") in self.param_names.flats
                else self.xp.zeros((len(d_arr), 1))
                for k in self.coord_names
            )
        )
        # log-likelihood
        lnliks = self.xp.clip(  # FIXME! higher order, not clip.
            1 / self._bma
            + (ms * (0.5 - d_arr / self._bma))
            + (ms**2 / 2 * (self._bma / 6 - d_arr + d_arr**2 / self._bma))
            + (
                (ms**3 * (2 * d_arr - self._bma) * d_arr * (self._bma - d_arr))
                / (12 * self._bma)
            ),
            0
        )
        lnliks = self.xp.log(liks)

        return ln_wgt + (indicator * lnliks).sum(1, keepdim=True)

    # ========================================================================
    # ML

    def forward(self, data: Data[Array]) -> Array:
        """Forward pass.

        Parameters
        ----------
        data : Data[Array]
            Input.

        Returns
        -------
        Array
            fraction, mean, sigma
        """
        # The forward step runs on the normalized coordinates
        data = self.data_scaler.transform(data, names=self.data_scaler.names)
        pred = self.xp.hstack(
            (
                self.xp.zeros((len(data), 1)),  # add the weight
                self.net(data[:, self.indep_coord_names, 0]),
            )
        )
        return self._forward_priors(pred, data)
