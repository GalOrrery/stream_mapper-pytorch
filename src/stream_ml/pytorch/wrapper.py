"""Core feature."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Any

from stream_ml.pytorch._base import ModelBase

if TYPE_CHECKING:
    from torch import nn

    from stream_ml.core.data import Data
    from stream_ml.core.params.core import Params
    from stream_ml.pytorch.typing import Array

__all__: list[str] = []


@dataclass(unsafe_hash=True)
class WithWeightModel(ModelBase):
    """Model with a weight.

    Parameters
    ----------
    net : NNField[NNModel], keyword-only
        The neural network.

    array_namespace : ArrayNamespace[Array], keyword-only
        The array namespace.

    weight : `torch.nn.Module`, keyword-only
        The weight network. This should be a `torch.nn.Module` that takes some
        inputs and returns one output.
    weight_coords : tuple[str, ...], keyword-only
        The coordinates that are used as inputs to the weight network.

    coord_names : tuple[str, ...], keyword-only
        The names of the coordinates, not including the 'independent' variable.
        E.g. for independent variable 'phi1' this might be ('phi2', 'prlx',
        ...).
    coord_bounds : Mapping[str, tuple[float, float]], keyword-only
        The bounds on the coordinates. If not provided, the bounds are (-inf,
        inf) for all coordinates.

    param_names : `~stream_ml.core.params.ParamNames`, keyword-only
        The names of the parameters. Parameters dependent on the coordinates are
        grouped by the coordinate name. E.g. ('weight', ('phi1', ('mu',
        'sigma'))).
    param_bounds : `~stream_ml.core.params.ParamBounds`, keyword-only
        The bounds on the parameters.

    name : str or None, optional keyword-only
        The (internal) name of the model, e.g. 'stream' or 'background'. Note
        that this can be different from the name of the model when it is used in
        a mixture model (see :class:`~stream_ml.core.core.MixtureModel`).
    """

    _: KW_ONLY
    weight: nn.Module
    weight_coords: tuple[str, ...] = ("phi1",)

    def forward(self, data: Data[Array]) -> Array:
        """Forward pass.

        Parameters
        ----------
        data : Data[Array]
            Input.

        Returns
        -------
        Array
        """
        # The forward step runs on the normalized coordinates
        data = self.data_scaler.transform(data, names=self.data_scaler.names)
        w = self.weight(data[:, self.weight_coords, 0])
        return self._forward_priors(self.xp.hstack((w, self.net(data))), data)

    def ln_likelihood(
        self, mpars: Params[Array], data: Data[Array], **kwargs: Any
    ) -> Array:
        """Log-likelihood.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array]
            Data.
        **kwargs : Any
            Keyword arguments.
        """
        return self.xp.log(mpars[("weight",)]) + self.net.ln_likelihood(
            mpars, data, **kwargs
        )
