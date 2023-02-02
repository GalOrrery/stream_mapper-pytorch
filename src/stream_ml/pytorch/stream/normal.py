"""Core feature."""

from __future__ import annotations

# STDLIB
import functools
import operator
from dataclasses import KW_ONLY, InitVar, dataclass
from math import inf
from typing import TYPE_CHECKING

# THIRD-PARTY
import torch as xp
from torch import nn
from torch.distributions.normal import Normal as TorchNormal

# LOCAL
from stream_ml.core.api import WEIGHT_NAME
from stream_ml.core.data import Data
from stream_ml.core.params import ParamBounds, ParamNames, Params
from stream_ml.core.params.names import ParamNamesField
from stream_ml.core.prior.bounds import NoBounds
from stream_ml.core.typing import BoundsT, ArrayNamespace
from stream_ml.core.utils.frozen_dict import FrozenDict
from stream_ml.pytorch.base import ModelBase
from stream_ml.pytorch.prior.bounds import PriorBounds, SigmoidBounds

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.pytorch.typing import Array

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

    n_features: int = 50
    n_layers: int = 3

    _: KW_ONLY
    array_namespace: InitVar[ArrayNamespace]

    param_names: ParamNamesField = ParamNamesField(
        (WEIGHT_NAME, (..., ("mu", "sigma")))
    )

    def __post_init__(self, array_namespace: ArrayNamespace) -> None:
        super().__post_init__(array_namespace=array_namespace)

        # Validate the coord_names
        if len(self.coord_names) != 1:
            msg = "Only one coordinate is supported, e.g ('phi2',)"
            raise ValueError(msg)

        # Define the layers of the neural network:
        # Total: in (phi) -> out (fraction, mean, sigma)
        self.layers = nn.Sequential(
            nn.Linear(1, self.n_features),
            nn.Tanh(),
            *functools.reduce(
                operator.add,
                (
                    (nn.Linear(self.n_features, self.n_features), nn.Tanh())
                    for _ in range(self.n_layers - 2)
                ),
            ),
            nn.Linear(self.n_features, 3),
        )

    @classmethod
    def from_simpler_inputs(
        cls,
        n_features: int = 50,
        n_layers: int = 3,
        *,
        coord_name: str,
        coord_bounds: BoundsT = (-inf, inf),
        weight_bounds: PriorBounds | BoundsT = SigmoidBounds(_eps, 1),  # noqa: B_eps008
        mu_bounds: PriorBounds | BoundsT | None | NoBounds = None,
        sigma_bounds: PriorBounds | BoundsT = SigmoidBounds(_eps, 0.3),  # noqa: B008
    ) -> Normal:
        """Create a Normal from a simpler set of inputs.

        Parameters
        ----------
        n_features : int, optional
            Number of features, by default 50.
        n_layers : int, optional
            Number of layers, by default 3.

        coord_name : str, keyword-only
            Coordinate name.
        coord_bounds : BoundsT, optional keyword-only
            Coordinate bounds.
        weight_bounds : PriorBounds | BoundsT, optional keyword-only
            Bounds on the mixture parameter.
        mu_bounds : PriorBounds | BoundsT, optional keyword-only
            Bounds on the mean.
        sigma_bounds : PriorBounds | BoundsT, optional keyword-only
            Bounds on the standard deviation.

        Returns
        -------
        Normal
        """
        return cls(
            n_features=n_features,
            n_layers=n_layers,
            coord_names=(coord_name,),
            param_names=ParamNames((WEIGHT_NAME, (coord_name, ("mu", "sigma")))),  # type: ignore[arg-type] # noqa: E501
            coord_bounds=FrozenDict({coord_name: coord_bounds}),  # type: ignore[arg-type] # noqa: E501
            param_bounds=ParamBounds(  # type: ignore[arg-type]
                {
                    WEIGHT_NAME: cls._make_bounds(weight_bounds, (WEIGHT_NAME,)),
                    coord_name: FrozenDict(
                        mu=cls._make_bounds(mu_bounds, (coord_name, "mu")),
                        sigma=cls._make_bounds(sigma_bounds, (coord_name, "sigma")),
                    ),
                }
            ),
        )

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

    def ln_prior_arr(self, mpars: Params[Array], data: Data[Array]) -> Array:
        """Log prior.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array]
            Data.

        Returns
        -------
        Array
        """
        lnp = xp.zeros_like(mpars[(WEIGHT_NAME,)])  # 100%
        # Bounds
        lnp += self._ln_prior_coord_bnds(mpars, data)
        for bound in self.param_bounds.flatvalues():
            lnp += bound.logpdf(mpars, data, self, lnp)

        # TODO: use super().ln_prior_arr(mpars, data, current_lnp) once
        #       the last argument is added to the signature.
        for prior in self.priors:
            lnp += prior.logpdf(mpars, data, self, lnp)

        return lnp

    # ========================================================================
    # ML

    def forward(self, data: Data[Array]) -> Array:
        """Forward pass.

        Parameters
        ----------
        data : Data[Array]
            Input. Only uses the first argument.

        Returns
        -------
        Array
            fraction, mean, sigma
        """
        nn = self._forward_prior(self.layers(data[self.indep_coord_name]), data)

        # Call the prior to limit the range of the parameters
        for prior in self.priors:
            nn = prior(nn, data, self)

        return nn
