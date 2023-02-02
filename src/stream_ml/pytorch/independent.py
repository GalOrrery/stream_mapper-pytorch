"""Core feature."""

from __future__ import annotations

# STDLIB
from dataclasses import dataclass

# LOCAL
from stream_ml.core.independent import IndependentModels as CoreIndependentModels
from stream_ml.pytorch.bases import ModelsBase
from stream_ml.pytorch.typing import Array

__all__: list[str] = []


# TODO: better mro
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
