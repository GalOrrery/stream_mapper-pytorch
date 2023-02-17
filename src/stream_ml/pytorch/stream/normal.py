"""Gaussian stream model."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

import torch as xp
from torch import nn
from torch.distributions.normal import Normal as TorchNormal

from stream_ml.core.params.bounds import ParamBoundsField
from stream_ml.core.params.names import ParamNamesField
from stream_ml.core.setup_package import WEIGHT_NAME
from stream_ml.pytorch.base import ModelBase
from stream_ml.pytorch.prior.bounds import SigmoidBounds
from stream_ml.pytorch.typing import Array

if TYPE_CHECKING:
    from stream_ml.core.data import Data
    from stream_ml.core.params import Params
    from stream_ml.core.typing import ArrayNamespace

__all__: list[str] = []


_eps = float(xp.finfo(xp.float32).eps)


@dataclass(unsafe_hash=True)
class Normal(ModelBase):
    r"""2D Gaussian with mixture weight.

    :math:`(weight, \mu, \sigma)(\phi1)`

    Parameters
    ----------
    n_layers : int, optional
        Number of hidden layers, by default 3.
    hidden_features : int, optional
        Number of hidden features, by default 50.
    sigma_upper_limit : float, optional keyword-only
        Upper limit on sigma, by default 0.3.
    fraction_upper_limit : float, optional keyword-only
        Upper limit on fraction, by default 0.45.s
    """

    _: KW_ONLY
    param_names: ParamNamesField = ParamNamesField(
        (WEIGHT_NAME, (..., ("mu", "sigma")))
    )
    param_bounds: ParamBoundsField[Array] = ParamBoundsField[Array](
        {  # reasonable guess for parameter bounds
            WEIGHT_NAME: SigmoidBounds(1e-10, 0.5),
            ...: {"mu": SigmoidBounds(-5.0, 5.0), "sigma": SigmoidBounds(0.05, 1.5)},
        }
    )

    def __post_init__(
        self, array_namespace: ArrayNamespace[Array], net: nn.Module | None
    ) -> None:
        # Initialize the network
        if net is not None:
            nnet = net
        else:
            nnet = nn.Sequential(
                nn.Linear(1, 36),
                nn.Tanh(),
                nn.Linear(36, 36),
                nn.Tanh(),
                nn.Linear(36, 3),
            )

        super().__post_init__(array_namespace=array_namespace, net=nnet)

        # Validate the coord_names
        if len(self.coord_names) != 1:
            msg = "Only one coordinate is supported, e.g ('phi2',)"
            raise ValueError(msg)

    # ========================================================================
    # Statistics

    def ln_likelihood_arr(
        self, mpars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Log-likelihood of the stream.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array]
            Data (phi1, phi2).
        **kwargs : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        c = self.coord_names[0]
        eps = xp.finfo(mpars[(WEIGHT_NAME,)].dtype).eps  # TOOD: or tiny?
        lik = TorchNormal(mpars[c, "mu"], xp.clip(mpars[c, "sigma"], min=eps)).log_prob(
            data[c]
        )
        return xp.log(xp.clip(mpars[(WEIGHT_NAME,)], min=eps)) + lik
