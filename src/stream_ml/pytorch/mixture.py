"""Core feature."""

from __future__ import annotations

# STDLIB
from dataclasses import dataclass
from typing import Literal

# THIRD-PARTY
import torch as xp

# LOCAL
from stream_ml.core.data import Data
from stream_ml.core.mixture import MixtureModel as CoreMixtureModel
from stream_ml.core.params import Params
from stream_ml.pytorch.bases import ModelsBase
from stream_ml.pytorch.typing import Array

__all__: list[str] = []


@dataclass(unsafe_hash=True)
class MixtureModel(ModelsBase, CoreMixtureModel[Array]):
    """Full Model.

    Parameters
    ----------
    models : Mapping[str, Model], optional postional-only
        Mapping of Models. This allows for strict ordering of the Models and
        control over the type of the models attribute.
    **more_models : Model
        Additional Models.
    """

    def _hook_unpack_bkg_weight(
        self, weight: Array | Literal[1], mp_arr: Array
    ) -> Array:
        """Hook to unpack the background weight."""
        if isinstance(weight, int):
            weight = xp.zeros((len(mp_arr), 1), dtype=mp_arr.dtype)
        return xp.hstack((weight, mp_arr))

    # ===============================================================
    # Statistics

    def ln_likelihood_arr(
        self, mpars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Log likelihood.

        Just the log-sum-exp of the individual log-likelihoods.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array]
            Data.
        **kwargs : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        # Get the parameters for each model, stripping the model name,
        # and use that to evaluate the log likelihood for the model.
        liks = tuple(
            model.ln_likelihood_arr(
                mpars.get_prefixed(name),
                data,
                **self._get_prefixed_kwargs(name, kwargs),
            )
            for name, model in self.components.items()
        )
        # Sum over the models, keeping the data dimension
        return xp.logsumexp(xp.hstack(liks), dim=1, keepdim=True)
