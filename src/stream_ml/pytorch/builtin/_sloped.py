"""Built-in background models."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, replace
from typing import TYPE_CHECKING

from stream_ml.core.params.bounds import ParamBoundsField
from stream_ml.core.params.names import ParamNamesField
from stream_ml.core.setup_package import WEIGHT_NAME
from stream_ml.core.utils.frozen_dict import FrozenDict
from stream_ml.core.utils.scale.utils import scale_params
from stream_ml.pytorch.base import ModelBase
from stream_ml.pytorch.prior.bounds import SigmoidBounds
from stream_ml.pytorch.typing import Array, NNModel

__all__: list[str] = []

if TYPE_CHECKING:
    from stream_ml.core.data import Data
    from stream_ml.core.params import Params


@dataclass(unsafe_hash=True)
class Sloped(ModelBase):
    r"""Tilted separately in each dimension.

    In each dimension the background is a sloped straight line between points
    ``a`` and ``b``. The slope is ``m``.

    The non-zero portion of the PDF, where :math:`a < x < b` is

    .. math::

        f(x) = m(x - \frac{a + b}{2}) + \frac{1}{b-a}

    Parameters
    ----------
    net : nn.Module, keyword-only
        The network to use. If not provided, a new one will be created. Must be
        a layer with 1 input and ``len(param_names)-1`` outputs.
    """

    _: KW_ONLY
    param_names: ParamNamesField = ParamNamesField(
        (WEIGHT_NAME, (..., ("slope",))), requires_all_coordinates=False
    )
    param_bounds: ParamBoundsField[Array] = ParamBoundsField[Array](
        {
            WEIGHT_NAME: SigmoidBounds(1e-10, 1.0, param_name=(WEIGHT_NAME,)),
            ...: {"slope": SigmoidBounds(-1.0, 1.0)},  # param_name is filled in later
        }
    )
    require_mask: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()

        _bma = []  # Pre-compute the associated constant factors
        # Add the slope param_names to the coordinate bounds
        # TODO! instead un-freeze then
        # re-freeze.
        for k, (a, b) in self.coord_bounds.items():
            a_ = self.data_scaler.transform(a, names=(k,))
            b_ = self.data_scaler.transform(b, names=(k,))

            if k in self.param_names.top_level:
                _bma.append(b_ - a_)

            bv = 2 / (b_ - a_) ** 2  # absolute value of the bound

            if k in self.param_bounds and isinstance(self.param_bounds[k], FrozenDict):
                pb = self.param_bounds[k, "slope"]
                # Mutate the underlying dictionary
                self.param_bounds[k]._dict["slope"] = replace(
                    pb, lower=-max(pb.lower, bv), upper=min(pb.upper, bv)
                )

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
        data = self.data_scaler.transform(data, names=self.data_scaler.names)
        mpars = scale_params(self, mpars)

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

        # Compute the log-likelihood, columns are coordinates.
        ln_lks = self.xp.zeros((len(ln_wgt), len(self.coord_bounds)))
        for i, (k, (a, b)) in enumerate(self.coord_bounds.items()):
            a_ = self.data_scaler.transform(a, names=(k,))
            b_ = self.data_scaler.transform(b, names=(k,))
            # Get the slope from `mpars` we check param_names to see if the
            # slope is a parameter. If it is not, then we assume it is 0.
            # When the slope is 0, the log-likelihood reduces to a Uniform.
            m = mpars[(k, "slope")] if (k, "slope") in self.param_names.flats else 0
            ln_lks[:, i] = self.xp.log(
                m * (data[k][:, 0] - (a_ + b_) / 2) + 1 / (b_ - a_)
            )

        return ln_wgt + (indicator * ln_lks).sum(1, keepdim=True)

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
                self.xp.zeros((len(data), 1)),  # weight placeholder
                (self.net(data[:, self.indep_coord_names, 0]) - 0.5) / self._bma,
            )
        )
        return self._forward_priors(pred, data)
