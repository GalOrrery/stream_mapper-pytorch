"""Core feature."""

from __future__ import annotations

# STDLIB
from dataclasses import KW_ONLY, InitVar, dataclass

# THIRD-PARTY
import torch as xp

# LOCAL
from stream_ml.core.data import Data
from stream_ml.core.params import Params
from stream_ml.pytorch.typing import Array
from stream_ml.pytorch.base import ModelBase
from nflows.flows.base import Flow

__all__: list[str] = []


@dataclass(unsafe_hash=True)
class FlowModel(ModelBase):

    model: InitVar[Flow]
    _: KW_ONLY
    with_grad: bool = True

    def __post_init__(self, model) -> None:
        super().__post_init__()
        self.wrapped = model

    def ln_likelihood_arr(
        self, mpars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        if not self.with_grad:
            with xp.no_grad():
                return self.wrapped.log_prob(data[self.coord_names].array)[:, None]

        return self.wrapped.log_prob(data[self.coord_names].array)[:, None]

    def ln_prior_arr(self, mpars: Params[Array], data: Data[Array]) -> Array:
        return xp.zeros((len(data), 1))

    def forward(self, data: Data[Array]) -> Array:
        return xp.asarray([])
