"""Built-in background models."""

from __future__ import annotations

__all__: tuple[str, ...] = ()

from dataclasses import KW_ONLY, dataclass, replace
from typing import TYPE_CHECKING

from stream_mapper.core.params import scale_params
from stream_mapper.core.utils import names_intersect
from stream_mapper.core.utils.frozen_dict import FrozenDict

from stream_mapper.pytorch._base import ModelBase

if TYPE_CHECKING:
    from stream_mapper.core import Data, Params

    from stream_mapper.pytorch.typing import Array


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
        a layer with 1 input and ``len(param names)-1`` outputs.

    """

    _: KW_ONLY
    require_mask: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()

        _bma = []  # Pre-compute the associated constant factors
        # Add the slope param_names to the coordinate bounds
        # TODO! instead un-freeze then
        # re-freeze.
        for k, (a, b) in self.coord_bounds.items():
            a_: Array = self.data_scaler.transform(a, names=(k,), xp=self.xp)
            b_: Array = self.data_scaler.transform(b, names=(k,), xp=self.xp)

            if k in self.params:
                _bma.append(b_ - a_)

            bv = 2 / (b_ - a_) ** 2  # absolute value of the bound

            if k in self.params and isinstance(self.params[k], FrozenDict):
                pb = self.params[k, "slope"].bounds
                # Mutate the underlying dictionary
                object.__setattr__(
                    self.params[k, "slope"],
                    "bounds",
                    replace(pb, lower=-max(pb.lower, bv), upper=min(pb.upper, bv)),
                )

        self._bma = self.xp.asarray(_bma)

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
        data = self.data_scaler.transform(
            data, names=names_intersect(data.names, self.data_scaler.names), xp=self.xp
        )
        mpars = scale_params(self, mpars)

        # 'where' is used to indicate which data points are available. If
        # 'where' is not provided, then all data points are assumed to be
        # available.
        if mask is not None:
            indicator = mask[tuple(self.coord_bounds.keys())].array
        elif self.require_mask:
            msg = "mask is required"
            raise ValueError(msg)
        else:
            indicator = self.xp.ones((len(data), 1), dtype=int)
            # This has shape (N, 1) so will broadcast correctly.

        # Compute the log-likelihood, columns are coordinates.
        ln_lks = self.xp.zeros((len(data), len(self.coord_bounds)))
        for i, (k, (a, b)) in enumerate(self.coord_bounds.items()):
            a_: Array = self.data_scaler.transform(a, names=(k,), xp=self.xp)
            b_: Array = self.data_scaler.transform(b, names=(k,), xp=self.xp)
            # Get the slope from `mpars` we check param_names to see if the
            # slope is a parameter. If it is not, then we assume it is 0.
            # When the slope is 0, the log-likelihood reduces to a Uniform.
            m = mpars[(k, "slope")] if (k, "slope") in self.params.flatskeys() else 0
            ln_lks[:, i] = self.xp.log(m * (data[k] - (a_ + b_) / 2) + 1 / (b_ - a_))

        return (indicator * ln_lks).sum(1)

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
            fraction, mean, ln-sigma

        """
        # The forward step runs on the normalized coordinates
        data = self.data_scaler.transform(
            data, names=names_intersect(data, self.data_scaler), xp=self.xp
        )
        pred = self.xp.hstack(
            (
                self.xp.zeros((len(data), 1)),  # weight placeholder
                (self.net(data[self.indep_coord_names].array) - 0.5) / self._bma,
            )
        )
        return self._forward_priors(pred, data)
