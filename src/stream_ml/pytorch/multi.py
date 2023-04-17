"""Core feature."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from torch import nn

from stream_ml.core import Model, NNField
from stream_ml.core.multi import IndependentModels as CoreIndependentModels
from stream_ml.core.multi import MixtureModel as CoreMixtureModel
from stream_ml.core.multi import ModelsBase as CoreModelsBase
from stream_ml.core.prior import PriorBase  # noqa: TCH001
from stream_ml.core.prior.bounds import NoBounds, PriorBounds
from stream_ml.core.setup_package import BACKGROUND_KEY
from stream_ml.core.utils.frozen_dict import FrozenDictField
from stream_ml.core.utils.sentinel import MISSING
from stream_ml.pytorch.prior.bounds import SigmoidBounds
from stream_ml.pytorch.typing import Array, NNModel

__all__: list[str] = []

if TYPE_CHECKING:
    from stream_ml.core.data import Data

    Self = TypeVar("Self", bound="MixtureModel")


@dataclass
class ModelsBase(nn.Module, CoreModelsBase[Array, NNModel]):
    """Multi-model base class."""

    components: FrozenDictField[str, Model[Array]] = FrozenDictField()

    _: KW_ONLY
    priors: tuple[PriorBase[Array], ...] = ()

    DEFAULT_PARAM_BOUNDS: ClassVar[PriorBounds] = NoBounds()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Register the models with pytorch.
        for name, model in self.components.items():
            self.add_module(name=name, module=model)


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

    priors : tuple[PriorBase, ...], optional keyword-only
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
            fraction, mean, sigma.
        """
        pred = self.xp.concatenate(
            [model(data) for model in self.components.values()], dim=1
        )

        # There's no need to call the parameter bounds prior here, since
        # the parameters are already constrained by each component.
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

    net: NNField[NNModel] = NNField(default=MISSING)
    _: KW_ONLY

    DEFAULT_PARAM_BOUNDS: ClassVar = SigmoidBounds(0, 1)

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
            fraction, mean, sigma.
        """
        # Predict the weights, except the background weight, which is
        # always 1 - sum(weights).
        scaled_data = self.data_scaler.transform(data, names=self.data_scaler.names)
        # TODO! need forward priors
        weights = self.net(scaled_data[:, self.indep_coord_names, 0])  # (N, K, ...)

        # Parameter bounds, skipping the background weight if present.
        for bnd in self.param_bounds.flatvalues()[: -int(self._includes_bkg) or None]:
            weights = bnd(weights, scaled_data, self)

        # Predict the parameters for each component.
        # The weight is added
        tuple(self.components.keys())
        preds = []
        for i, (name, model) in enumerate(self.components.items()):
            if name == BACKGROUND_KEY:
                weight = 1 - weights.sum(1, keepdims=True)
            else:
                weight = weights[:, i, None]

            preds.append(weight)
            preds.append(model(data))

        pred = self.xp.concatenate(preds, dim=1)

        # Other priors  # TODO: a better way to do the order of the priors.
        for prior in self.priors:
            pred = prior(pred, scaled_data, self)

        # go around again to ensure that the background weight is 1 - sum(weights)
        if self._includes_bkg:
            i = tuple(self.components.keys()).index(BACKGROUND_KEY)
            pred[:, i] = 1 - weights.sum(1)

        return pred
