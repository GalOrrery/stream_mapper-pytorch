"""Core feature."""

from __future__ import annotations

# STDLIB
from dataclasses import KW_ONLY, dataclass
from math import inf
from typing import TYPE_CHECKING

# THIRD-PARTY
import torch as xp

# LOCAL
from stream_ml.core.api import WEIGHT_NAME
from stream_ml.core.data import Data
from stream_ml.core.prior.base import PriorBase
from stream_ml.core.utils.funcs import within_bounds
from stream_ml.pytorch.typing import Array

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.core.api import Model
    from stream_ml.core.params.core import Params

__all__: list[str] = []


@dataclass(frozen=True)
class Lasso(PriorBase[Array]):
    """Lasso prior encouraging the parameter towards a number.

    TODO: It would be good if Prior hyperparameters could be ML parameters.
          The prior would need to inherit from nn.Module and be registered
          on the enclosing model.

    Parameters
    ----------
    lamda :
    """

    lamda: float = 0.005

    def logpdf(
        self,
        mpars: Params[Array],
        data: Data[Array],
        model: Model[Array],
        current_lnpdf: Array | None = None,
        /,
    ) -> Array | float:
        """Evaluate the logpdf."""
        ...


@dataclass(frozen=True)
class BoundedHardThreshold(PriorBase[Array]):
    """Threshold prior.

    Parameters
    ----------
    threshold : float, optional
        The threshold, by default 0.005
    lower : float, optional
        The lower bound in the domain of the prior, by default `-inf`.
    upper : float, optional
        The upper bound in the domain of the prior, by default `inf`.
    """

    threshold: float = 0.005
    _: KW_ONLY
    coord_name: str = "phi1"
    lower: float = -inf
    upper: float = inf

    def logpdf(
        self,
        mpars: Params[Array],
        data: Data[Array],
        model: Model[Array],
        current_lnpdf: Array | None = None,
        /,
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

        Returns
        -------
        Array
            The logpdf.
        """
        lnp = xp.zeros_like(mpars[(WEIGHT_NAME,)])
        lnp[
            within_bounds(data[self.coord_name], self.lower, self.upper)
            & (mpars[(WEIGHT_NAME,)] < self.threshold)
        ] = -inf
        return lnp

    def __call__(self, pred: Array, data: Data[Array], model: Model[Array], /) -> Array:
        """Evaluate the forward step in the prior.

        Parameters
        ----------
        pred : Array, position-only
            The input to evaluate the prior at.
        data : Data[Array], position-only
            The data to evaluate the prior at.
        model : `~stream_ml.core.Model`, position-only
            The model to evaluate the prior at.

        Returns
        -------
        Array
        """
        im1 = model.param_names.flat.index(WEIGHT_NAME)
        where = within_bounds(data[self.coord_name][:, 0], self.lower, self.upper)

        out = pred.clone()
        out[where, im1] = xp.threshold(pred[where, im1], self.threshold, 0)

        return out
