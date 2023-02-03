"""Track priors.

.. todo::

    - Add a ControlRegions prior that is an equiprobability region centered on a
      point. This is a lot less informative than the ControlPoints.

"""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

from stream_ml.core.data import Data
from stream_ml.core.prior.base import PriorBase
from stream_ml.core.typing import ArrayNamespace
from stream_ml.pytorch.typing import Array

if TYPE_CHECKING:
    from stream_ml.core.api import Model
    from stream_ml.core.params.core import Params

__all__: list[str] = []


#####################################################################


@dataclass(frozen=True)
class ControlPoints(PriorBase[Array]):
    """Control points prior.

    Parameters
    ----------
    control_points : Data[Array]
        The control points.
    lamda : float, optional
        Importance hyperparameter.
    """

    control_points: Data[Array]
    lamda: float = 0.05  # TODO? as a trainable Parameter.
    _: KW_ONLY
    coord_name: str = "phi1"
    component_param_name: str = "mu"

    def __post_init__(self) -> None:
        """Post-init."""
        # Pre-store the control points, seprated by indep & dep parameters.
        self._cpts_indep: Data[Array]
        object.__setattr__(self, "_cpts_indep", self.control_points[(self.coord_name,)])

        dep_names = tuple(n for n in self.control_points.names if n != self.coord_name)
        self._control_point_deps: Data[Array]
        object.__setattr__(self, "_control_point_deps", self.control_points[dep_names])

        super().__post_init__()

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
        cmpars = model.unpack_params_from_arr(model(self._cpts_indep))
        cmp_arr = xp.hstack(  # (C, F)
            tuple(
                cmpars[(n, self.component_param_name)]
                for n in self._control_point_deps.names
            )
        )

        # For each control point, add the squared distance to the logpdf.
        return (
            -self.lamda
            * ((cmp_arr - self._control_point_deps.array) ** 2).sum()  # (C, F) -> 1
        )


#####################################################################


@dataclass(frozen=True)
class ControlRegions(PriorBase[Array]):
    r"""Control regions prior.

    A unit box convolved with a higher-order Gaussian. For mean :math:`\\mu`,
    width :math:`w`, and power :math:`P` the PDF is

    .. math::

        PDF(x, \\mu, w, P) = \\erf{P ((x - \\mu) + w0)} - \\erf{P ((x - \\mu) - w0)}

    Parameters
    ----------
    control_points : Data[Array]
        The control points.
    lamda : float, optional
        Importance hyperparameter.
    width : float, optional
        Width of the region.
    flattening : float, optional
        The power of the super-Gaussian. Higher powers are flatter.
    """

    control_points: Data[Array]
    lamda: float = 0.05  # TODO? as a trainable Parameter.
    width: float = 0.5
    flattening: float = 5
    _: KW_ONLY
    coord_name: str = "phi1"
    component_param_name: str = "mu"

    def __post_init__(self) -> None:
        """Post-init."""
        # Adjust the flattening to be adjusted by the width.
        object.__setattr__(self, "flattening", self.flattening / self.width)

        # Pre-store the control points, seprated by indep & dep parameters.
        self._cpts_indep: Data[Array]
        object.__setattr__(self, "_cpts_indep", self.control_points[(self.coord_name,)])

        dep_names = tuple(n for n in self.control_points.names if n != self.coord_name)
        self._control_point_deps: Data[Array]
        object.__setattr__(self, "_control_point_deps", self.control_points[dep_names])

        super().__post_init__()

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
        cmpars = model.unpack_params_from_arr(model(self._cpts_indep))
        cmp_arr = xp.hstack(  # (C, F)
            tuple(
                cmpars[(n, self.component_param_name)]
                for n in self._control_point_deps.names
            )
        )

        # Unnormalized PDF.
        pdf = xp.special.erf(
            self.flattening * ((cmp_arr - self._control_point_deps.array) + self.width)
        ) - xp.special.erf(
            self.flattening * ((cmp_arr - self._control_point_deps.array) - self.width)
        )

        # For each control region, add the distance to the logpdf.
        return (
            self.lamda
            * xp.log(
                xp.clip(pdf / 2, min=xp.finfo(cmp_arr.dtype).tiny)
            ).sum()  # (C, F) -> 1
        )
