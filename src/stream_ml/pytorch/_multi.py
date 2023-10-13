"""Core feature."""

from __future__ import annotations

__all__: tuple[str, ...] = ()

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from torch import nn

from stream_ml.core import BACKGROUND_KEY, NNField
from stream_ml.core import IndependentModels as CoreIndependentModels
from stream_ml.core import MixtureModel as CoreMixtureModel
from stream_ml.core import ModelsBase as CoreModelsBase
from stream_ml.core._api import SupportsXPNN
from stream_ml.core.utils import names_intersect
from stream_ml.core.utils.sentinel import MISSING

from stream_ml.pytorch.typing import Array, NNModel

if TYPE_CHECKING:
    from stream_ml.core import Data

    Self = TypeVar("Self", bound="MixtureModel")


@dataclass
class ModelsBase(nn.Module, CoreModelsBase[Array, NNModel]):
    """Multi-model base class."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # Register the models with pytorch.
        for name, model in self.components.items():
            self.add_module(name=name, module=model)

    __setstate__ = SupportsXPNN.__setstate__


@dataclass(unsafe_hash=True)
class IndependentModels(ModelsBase, CoreIndependentModels[Array, NNModel]):
    """Composite of a few models that acts like one model.

    This is different from a mixture model in that the components are not
    separate, but are instead combined into a single model. Practically, this
    means:

    - All the components have the same weight.
    - The log-likelihoood of the composite model is the sum of the
      log-likelihooods of the components, not the log-sum-exp.

    Parameters
    ----------
    components : Mapping[str, Model], optional postional-only
        Mapping of Models. This allows for strict ordering of the Models and
        control over the type of the models attribute.

    name : str or None, optional keyword-only
        The (internal) name of the model, e.g. 'stream' or 'background'. Note
        that this can be different from the name of the model when it is used in
        a mixture model (see :class:`~stream_ml.core.IndependentModels`).

    priors : tuple[Prior, ...], optional keyword-only
        Mapping of parameter names to priors. This is useful for setting priors
        on parameters across models, e.g. the background and stream models in a
        mixture model.
    """

    def __post_init__(self) -> None:
        nn.Module.__init__(self)  # Needed for PyTorch
        super().__post_init__()

    def forward(self, data: Data[Array], /) -> Array:
        """Forward pass.

        Parameters
        ----------
        data : Data
            Input.

        Returns
        -------
        Array
            fraction, mean, ln-sigma.
        """
        pred = self.xp.concatenate(
            tuple(model(data) for model in self.components.values()), dim=1
        )
        # There are no prameter bounds for the independent models.
        # TODO: a better way to do the order of the priors.
        for prior in self.priors:
            pred = prior(pred, data, self)

        return pred


@dataclass(unsafe_hash=True)
class MixtureModel(ModelsBase, CoreMixtureModel[Array, NNModel]):
    """Full Model.

    Parameters
    ----------
    components : Mapping[str, Model], optional postional-only
        Mapping of Models. This allows for strict ordering of the Models and
        control over the type of the models attribute.
    net : NNModel, optional postional-only
        The neural network that is used to combine the components.
    """

    net: NNField[NNModel, NNModel] = NNField(default=MISSING)

    def __new__(cls: type[Self], *args: Any, **kwargs: Any) -> Self:  # noqa: ARG003
        """Initialize the model. This is needed for PyTorch."""
        self: Self = super().__new__(cls)
        # PyTorch needs to be initialized before attributes are assigned.
        nn.Module.__init__(self)
        return self

    def forward(self, data: Data[Array], /) -> Array:
        """Forward pass.

        Parameters
        ----------
        data : Data
            Input.

        Returns
        -------
        Array
            fraction, mean, ln-sigma.
        """
        # Predict the weights, except the background weight, which is
        # always 1 - sum(weights).
        scaled_data = self.data_scaler.transform(
            data, names=names_intersect(data, self.data_scaler), xp=self.xp
        )
        # TODO! need forward priors
        ln_weights = self.net(scaled_data[self.indep_coord_names].array)  # (N, K, ...)

        # Parameter bounds, skipping the background weight (if present),
        # since the Mixture NN should not predict it.
        for param in self.params.flatvalues()[self._bkg_slc]:
            ln_weights = param.bounds(ln_weights, scaled_data, self)

        # Predict the parameters for each component.
        # The weight is added
        preds: list[Array] = []
        wgt_is: list[int] = [-1] * len(self.components)
        counter: int = 0
        for i, (name, model) in enumerate(self.components.items()):
            ln_weight = (  # (N, 1)
                ln_weights[:, i]
                if name != BACKGROUND_KEY
                else self.xp.log(
                    -self.xp.expm1(self.xp.special.logsumexp(ln_weights, 1))
                )
            )[:, None]
            wgt_is[i] = counter

            pred = model(data)
            preds.extend((ln_weight, pred))
            counter += 1 + (pred.shape[1] if len(pred.shape) > 1 else 0)

        out = self.xp.concatenate(preds, 1)

        # Other priors  # TODO: a better way to do the order of the priors.
        for prior in self.priors:
            out = prior(out, scaled_data, self)

        # Ensure that the background weight is 1 - sum(weights)
        if self._includes_bkg:
            out[:, wgt_is[-1]] = self.xp.log(
                -self.xp.expm1(self.xp.special.logsumexp(out[:, wgt_is[:-1]], 1))
            )

        return out
