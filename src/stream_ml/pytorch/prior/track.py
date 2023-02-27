"""Track priors."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

import torch as xp

from stream_ml.core.data import Data  # noqa: TCH001
from stream_ml.core.prior.base import PriorBase
from stream_ml.pytorch.typing import Array

if TYPE_CHECKING:
    from stream_ml.core.api import Model
    from stream_ml.core.params.core import Params
    from stream_ml.core.typing import ArrayNamespace

__all__: list[str] = []


#####################################################################


@dataclass(frozen=True)
class TrackPriorBase(PriorBase[Array]):
    """Track Prior Base."""

    control_points: Data[Array]
    lamda: float = 0.05  # TODO? as a trainable Parameter.
    _: KW_ONLY
    coord_name: str = "phi1"
    component_param_name: str = "mu"

    def __post_init__(self) -> None:
        """Post-init."""
        super().__post_init__()

        # Pre-store the control points, seprated by indep & dep parameters.
        self._x: Data[Array]
        object.__setattr__(self, "_x", self.control_points[(self.coord_name,)])

        dep_names: tuple[str, ...] = tuple(
            n for n in self.control_points.names if n != self.coord_name
        )
        object.__setattr__(self, "_y_names", dep_names)

        self._y: Array
        object.__setattr__(
            self, "_y", xp.atleast_2d(xp.squeeze(self.control_points[dep_names].array))
        )


@dataclass(frozen=True)
class ControlPoints(TrackPriorBase):
    """Control points prior.

    Parameters
    ----------
    control_points : Data[Array]
        The control points.
    lamda : float, optional
        Importance hyperparameter.
    """

    def logpdf(
        self,
        mpars: Params[Array],
        data: Data[Array],
        model: Model[Array],
        current_lnpdf: Array | None = None,
        /,
        *,
        xp: ArrayNamespace[Array],
    ) -> Array | float:
        """Evaluate the logpdf.

        This log-pdf is added to the current logpdf. So if you want to set the
        logpdf to a specific value, you can uses the `current_lnpdf` to set the
        output value such that ``current_lnpdf + logpdf = <want>``.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array], position-only
            The data for which evaluate the prior.
        model : Model, position-only
            The model for which evaluate the prior.
        current_lnpdf : Array | None, optional position-only
            The current logpdf, by default `None`. This is useful for setting
            the additive log-pdf to a specific value.

        xp : ArrayNamespace[Array], keyword-only
            The array namespace.

        Returns
        -------
        Array
            The logpdf.
        """
        # Get the model parameters evaluated at the control points. shape (C, 1).
        cmpars = model.unpack_params_from_arr(model(self._x))
        cmp_arr = xp.hstack(  # (C, F)
            tuple(cmpars[(n, self.component_param_name)] for n in self._y_names)
        )

        # For each control point, add the squared distance to the logpdf.
        return -self.lamda * ((cmp_arr - self._y) ** 2).sum()  # (C, F) -> 1


#####################################################################


@dataclass(frozen=True)
class ControlRegions(TrackPriorBase):
    r"""Control regions prior.

    The gaussian control points work very well, but they are very informative.
    This prior is less informative, but still has a similar effect.
    It is a Gaussian, split at the peak, with a flat region in the middle.
    The split is done when the 1st derivative is 0, so it is smooth up to the
    1st derivative.

    .. math::

        \ln p(x, \mu, w) = \begin{cases}
            (x - (mu - w))^2 & x \leq mu - w \\
            0                & mu - w < x < mu + w \\
            (x - (mu + w))^2 & x \geq mu + w \\

    Parameters
    ----------
    control_points : Data[Array]
        The control points. These are the means of the regions (mu in the above).
    lamda : float, optional
        Importance hyperparameter.
        TODO: make this also able to be an array, so that each region can have
        a different width.
    width : float, optional
        Width of the region.
        TODO: make this also able to be an array, so that each region can have
        a different width.
    """

    width: float | Data[Array] = 0.5

    def __post_init__(self) -> None:
        """Post-init."""
        super().__post_init__()

        # Pre-store the width.
        self._w: Array
        object.__setattr__(
            self,
            "_w",
            xp.atleast_2d(xp.squeeze(self.width[self._y_names].array))
            if not isinstance(self.width, float)
            else xp.ones_like(self._y) * self.width,
        )

    def logpdf(
        self,
        mpars: Params[Array],
        data: Data[Array],
        model: Model[Array],
        current_lnpdf: Array | None = None,
        /,
        *,
        xp: ArrayNamespace[Array],
    ) -> Array | float:
        """Evaluate the logpdf.

        This log-pdf is added to the current logpdf. So if you want to set the
        logpdf to a specific value, you can uses the `current_lnpdf` to set the
        output value such that ``current_lnpdf + logpdf = <want>``.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array], position-only
            The data for which evaluate the prior.
        model : Model, position-only
            The model for which evaluate the prior.
        current_lnpdf : Array | None, optional position-only
            The current logpdf, by default `None`. This is useful for setting
            the additive log-pdf to a specific value.

        xp : ArrayNamespace[Array], keyword-only
            The array namespace.

        Returns
        -------
        Array
            The logpdf.
        """
        # Get model parameters evaluated at the control points. shape (C, 1).
        cmpars = model.unpack_params_from_arr(model(self._x))
        cmp_arr = xp.hstack(  # (C, F)
            tuple(cmpars[(n, self.component_param_name)] for n in self._y_names)
        )

        pdf = xp.zeros_like(cmp_arr)
        where = cmp_arr <= self._y - self._w
        pdf[where] = (cmp_arr[where] - (self._y[where] - self._w[where])) ** 2
        where = cmp_arr >= self._y + self._w
        pdf[where] = (cmp_arr[where] - (self._y[where] + self._w[where])) ** 2

        return -self.lamda * pdf.sum()  # (C, F) -> 1
