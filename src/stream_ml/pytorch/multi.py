"""Core feature."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from math import inf
from typing import TYPE_CHECKING, ClassVar

from torch import nn

from stream_ml.core.api import Model  # noqa: TCH001
from stream_ml.core.multi.bases import ModelsBase as CoreModelsBase
from stream_ml.core.multi.independent import IndependentModels as CoreIndependentModels
from stream_ml.core.multi.mixture import MixtureModel as CoreMixtureModel
from stream_ml.core.prior.base import PriorBase  # noqa: TCH001
from stream_ml.core.prior.bounds import PriorBounds  # noqa: TCH001
from stream_ml.core.utils.frozen_dict import FrozenDictField
from stream_ml.pytorch.prior.bounds import SigmoidBounds
from stream_ml.pytorch.typing import Array

__all__: list[str] = []

if TYPE_CHECKING:
    from stream_ml.core.data import Data


@dataclass
class ModelsBase(nn.Module, CoreModelsBase[Array]):
    """Multi-model base class."""

    components: FrozenDictField[str, Model[Array]] = FrozenDictField()

    _: KW_ONLY
    priors: tuple[PriorBase[Array], ...] = ()

    DEFAULT_BOUNDS: ClassVar[PriorBounds] = SigmoidBounds(-inf, inf)

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
        pred = self.xp.concatenate(
            [model(data) for model in self.components.values()], dim=1
        )

        # There's no need to call the parameter bounds prior here, since
        # the parameters are already constrained by each component.

        # TODO: a better way to do the order of the priors.
        # Call the prior to limit the range of the parameters.
        for prior in self.priors:
            pred = prior(pred, data, self)

        return pred


@dataclass(unsafe_hash=True)
class IndependentModels(ModelsBase, CoreIndependentModels[Array]):
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
        a mixture model (see :class:`~stream_ml.core.core.IndependentModels`).

    priors : tuple[PriorBase, ...], optional keyword-only
        Mapping of parameter names to priors. This is useful for setting priors
        on parameters across models, e.g. the background and stream models in a
        mixture model.
    """


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
