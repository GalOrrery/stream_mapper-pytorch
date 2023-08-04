"""Built-in background models."""

from __future__ import annotations

__all__: list[str] = []

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from stream_ml.pytorch import Data
    from stream_ml.pytorch.typing import Array, ArrayNamespace

# =============================================================================
# Cluster Mass Function


class StreamMassFunction(Protocol):
    """Stream Mass Function.

    Must be parametrized by gamma [0, 1], the normalized mass over the range of the
    isochrone.

    Returns the log-probability that stars of that mass (gamma) are in the
    population modeled by the isochrone.
    """

    def __call__(
        self, gamma: Array, x: Data[Array], *, xp: ArrayNamespace[Array]
    ) -> Array:
        r"""Log-probability of stars at position 'x' having mass 'gamma'.

        Parameters
        ----------
        gamma : Array[(F,))]
            The mass of the stars, normalized to [0, 1] over the range of the
            isochrone.
        x : Data[Array[(N,)]]
            The independent data. Normally this is :math:`\phi_1`.

        xp : ArrayNamespace[Array], keyword-only
            The array namespace.

        Returns
        -------
        Array[(N, F)]
        """
        ...


@dataclass(frozen=True)
class UniformStreamMassFunction(StreamMassFunction):
    def __call__(
        self, gamma: Array, x: Data[Array], *, xp: ArrayNamespace[Array]
    ) -> Array:
        return xp.zeros((len(x), len(gamma)))


@dataclass(frozen=True)
class HardCutoffMassFunction(StreamMassFunction):
    """Hard Cutoff IMF."""

    lower: float = 0
    upper: float = 1

    def __call__(
        self, gamma: Array, x: Data[Array], *, xp: ArrayNamespace[Array]
    ) -> Array:
        out = xp.full((len(x), len(gamma)), -xp.inf)
        out[:, (gamma >= self.lower) & (gamma <= self.upper)] = 0
        return out


@dataclass(frozen=True)
class StepwiseMassFunction(StreamMassFunction):
    """Hard Cutoff IMF."""

    boundaries: tuple[float, ...]  # (B + 1,)
    log_probs: tuple[float, ...]  # (B,)

    def __call__(
        self, gamma: Array, x: Data[Array], *, xp: ArrayNamespace[Array]
    ) -> Array:
        out = xp.full((len(x), len(gamma)), -xp.inf)
        for lower, upper, lnp in zip(
            self.boundaries[:-1], self.boundaries[1:], self.log_probs, strict=True
        ):
            out[:, (gamma >= lower) & (gamma < upper)] = lnp
        return out
