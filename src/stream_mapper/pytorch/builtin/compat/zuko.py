"""Core feature."""

from __future__ import annotations

__all__: tuple[str, ...] = ()

from dataclasses import dataclass
from typing import TYPE_CHECKING

from stream_mapper.pytorch.builtin.compat._flow import _FlowModel

if TYPE_CHECKING:
    from stream_mapper.pytorch import Data
    from stream_mapper.pytorch.typing import Array


@dataclass(unsafe_hash=True, repr=False)
class ZukoFlowModel(_FlowModel):
    """Normalizing flow model."""

    def _log_prob(self, data: Data[Array], idx: Array) -> Array:
        """Log-probability of the array."""
        return self.net(
            data[self.indep_coord_names].array[idx]
            if self.indep_coord_names is not None
            else None
        ).log_prob(data[self.coord_names].array[idx])
