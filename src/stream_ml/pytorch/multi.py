"""Core feature."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, ClassVar

from torch import nn

from stream_ml.core import Model  # noqa: TCH001
from stream_ml.core.multi import IndependentModels as CoreIndependentModels
from stream_ml.core.multi import MixtureModel as CoreMixtureModel
from stream_ml.core.multi import ModelsBase as CoreModelsBase
from stream_ml.core.prior import PriorBase  # noqa: TCH001
from stream_ml.core.prior.bounds import NoBounds, PriorBounds
from stream_ml.core.utils.frozen_dict import FrozenDictField
from stream_ml.pytorch.typing import Array, NNModel

__all__: list[str] = []

if TYPE_CHECKING:
    from stream_ml.core.data import Data


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
        nn.Module.__init__(self)  # Needed for PyTorch
        for name, model in self.components.items():
            self.add_module(name=name, module=model)

    # ========================================================================
    # ML

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
        # The 0-th element is the weight.
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
        # The 0-th element is the weight.
        # We need to cut out all other weights
        pred = self.xp.concatenate(
            [
                model(data)[:, int(self.has_weight[k] and i != 0) :]
                for i, (k, model) in enumerate(self.components.items())
            ],
            dim=1,
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
    models : Mapping[str, Model], optional postional-only
        Mapping of Models. This allows for strict ordering of the Models and
        control over the type of the models attribute.
    **more_models : Model
        Additional Models.
    """
