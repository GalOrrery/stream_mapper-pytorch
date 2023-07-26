"""Built-in background models."""

from __future__ import annotations

__all__: list[str] = []

from dataclasses import dataclass
from math import log
from typing import TYPE_CHECKING, ClassVar, Protocol

import torch as xp
from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs

if TYPE_CHECKING:
    from stream_ml.pytorch import Data
    from stream_ml.pytorch.typing import Array, ArrayNamespace


class StreamMassFunction(Protocol):
    """Stream Mass Function.

    Must be parametrized by gamma [0, 1], the normalized mass over the range of the
    isochrone.

    Returns the log-probability that stars of that mass (gamma) are in the
    population modeled by the isochrone.
    """

    def __call__(
        self, gamma: Array, x: Data[Array] | None, *, xp: ArrayNamespace[Array]
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
        self, gamma: Array, x: Data[Array] | None, *, xp: ArrayNamespace[Array]
    ) -> Array:
        return xp.zeros((1 if x is None else len(x), len(gamma)))


@dataclass(frozen=True)
class HardCutoffMassFunction(StreamMassFunction):
    """Hard Cutoff IMF."""

    lower: float = 0
    upper: float = 1

    def __call__(
        self, gamma: Array, x: Data[Array] | None, *, xp: ArrayNamespace[Array]
    ) -> Array:
        out = xp.full((1 if x is None else len(x), len(gamma)), -xp.inf)
        out[:, (gamma >= self.lower) & (gamma <= self.upper)] = 0
        return out


@dataclass(frozen=True)
class StepwiseMassFunction(StreamMassFunction):
    """Hard Cutoff IMF."""

    boundaries: tuple[float, ...]  # (B + 1,)
    log_probs: tuple[float, ...]  # (B,)

    def __call__(
        self, gamma: Array, x: Data[Array] | None, *, xp: ArrayNamespace[Array]
    ) -> Array:
        out = xp.full((1 if x is None else len(x), len(gamma)), -xp.inf)
        for lower, upper, lnp in zip(
            self.boundaries[:-1], self.boundaries[1:], self.log_probs, strict=True
        ):
            out[:, (gamma >= lower) & (gamma < upper)] = lnp
        return out


# --------------------------------------------------------------------------------


@dataclass(frozen=True)
class KroupaIMF(StreamMassFunction):
    """Kroupa IMF.

    https://arxiv.org/pdf/astro-ph/0102155.pdf

    Parameters
    ----------
    gamma_to_mass : :class:`torchcubicspline.NaturalCubicSpline`
        A callable that takes in gamma and returns the mass.
    """

    gamma_to_mass: NaturalCubicSpline

    ranges: ClassVar[tuple[float, ...]] = (0.01, 0.08, 0.5, 100)
    ln_ranges: ClassVar[tuple[float, ...]] = (1e-2, 8e-2, 5e-1, 1e2)

    def __post_init__(self) -> None:
        # TODO: need to normalize to gamma?
        # Compute the normalization by integrating over the mass range.
        self.ln_norm: float
        object.__setattr__(self, "ln_norm", 0)  # to avoid missing self-reference.
        gammas = xp.linspace(0, 1, 10_000)
        masses = self.gamma_to_mass.evaluate(gammas)[:, 0].to(dtype=gammas.dtype)
        lnpdfs = self(gammas, None, xp=xp)
        norm = float(xp.sum(xp.exp(lnpdfs)[:-1] * xp.diff(masses)))
        object.__setattr__(self, "ln_norm", log(norm))

    def __call__(
        self, gamma: Array, x: Data[Array] | None, *, xp: ArrayNamespace[Array]
    ) -> Array:
        out = xp.empty_like(gamma)
        mass = self.gamma_to_mass.evaluate(gamma)[:, 0].to(dtype=gamma.dtype)
        rng = xp.asarray(self.ranges, dtype=gamma.dtype)
        ln_rng = xp.asarray(self.ln_ranges, dtype=gamma.dtype)

        # https://arxiv.org/pdf/astro-ph/0102155.pdf
        if xp.any((mass < rng[0]) | (mass >= rng[-1])):
            msg = f"mass must be >= {rng[0]}."
            raise ValueError(msg)

        sel = (mass >= rng[0]) & (mass < rng[1])
        out[sel] = -0.3 * (xp.log(mass[sel]) - ln_rng[1])

        sel = (mass >= rng[1]) & (mass < rng[2])
        out[sel] = -1.3 * (xp.log(mass[sel]) - ln_rng[1])

        sel = (mass >= rng[2]) & (mass < rng[3])
        out[sel] = ln_rng[2] + 1.3 * ln_rng[1] - 2.3 * xp.log(mass[sel])

        return out - self.ln_norm

    # ===============================================================
    # Constructors

    @classmethod
    def from_arrays(cls, gamma: Array, mass: Array) -> KroupaIMF:
        """Construct from arrays."""
        coeffs = natural_cubic_spline_coeffs(gamma, mass[:, None])
        return cls(gamma_to_mass=NaturalCubicSpline(coeffs))
