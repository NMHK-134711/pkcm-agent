"""Deterministic randomness that lives inside the battle state.

Principle (b) in docs/DESIGN.md: the RNG is part of the state and is injected,
never global. That buys three things search cannot do without:

* exact replay -- a bug report is a single integer;
* "same position, different dice" rollouts, which is the whole premise of MCTS;
* the option to fold a roll into its expectation instead of sampling it.

The generator is SplitMix64, chosen because its entire state is one 64-bit
integer. Cloning a battle state therefore copies an int, not an object graph.

Two faces on the same generator:

``Rng``        frozen value. Every method returns ``(new_rng, result)``. This is
               what a ``BattleState`` stores.
``RngCursor``  mutable cursor, for use *inside* a single ``step()``. Threading a
               new ``Rng`` through every damage roll would drown the resolution
               code; a step opens a cursor, mutates it locally, and seals it back
               into the returned state. Purity holds at the step boundary, which
               is the boundary that matters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeVar

_MASK64 = (1 << 64) - 1
_GOLDEN_GAMMA = 0x9E3779B97F4A7C15

T = TypeVar("T")


def _mix(state: int) -> tuple[int, int]:
    """One SplitMix64 step: returns ``(next state, output)``."""
    state = (state + _GOLDEN_GAMMA) & _MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return state, z ^ (z >> 31)


class RngCursor:
    """Mutable view of an ``Rng``. Local to one ``step()``."""

    __slots__ = ("state",)

    def __init__(self, state: int) -> None:
        self.state = state

    def bits(self) -> int:
        self.state, value = _mix(self.state)
        return value

    def below(self, bound: int) -> int:
        """Uniform integer in ``[0, bound)``. Rejection-sampled, so unbiased."""
        if bound <= 0:
            raise ValueError(f"bound must be positive, got {bound}")
        if bound & (bound - 1) == 0:  # power of two: mask, no rejection needed
            return self.bits() & (bound - 1)
        limit = _MASK64 - (_MASK64 % bound)
        while True:
            value = self.bits()
            if value <= limit:
                return value % bound

    def between(self, low: int, high: int) -> int:
        """Uniform integer in ``[low, high]``, inclusive on both ends."""
        return low + self.below(high - low + 1)

    def chance(self, numerator: int, denominator: int) -> bool:
        """True with probability ``numerator / denominator``."""
        return self.below(denominator) < numerator

    def percent(self, chance: int) -> bool:
        return self.chance(chance, 100)

    def choice(self, options: Sequence[T]) -> T:
        return options[self.below(len(options))]

    def shuffled(self, options: Sequence[T]) -> list[T]:
        items = list(options)
        for i in range(len(items) - 1, 0, -1):
            j = self.below(i + 1)
            items[i], items[j] = items[j], items[i]
        return items

    def sample(self, options: Sequence[T], count: int) -> list[T]:
        """``count`` distinct elements, order randomized."""
        if count > len(options):
            raise ValueError(f"cannot sample {count} from {len(options)} options")
        return self.shuffled(options)[:count]

    def seal(self) -> "Rng":
        return Rng(self.state)


@dataclass(frozen=True, slots=True)
class Rng:
    """Immutable RNG value. Stored on the battle state; cloning copies an int."""

    state: int

    @classmethod
    def from_seed(cls, seed: int) -> "Rng":
        # Mix once so that adjacent seeds do not produce correlated streams.
        _, mixed = _mix(seed & _MASK64)
        return cls(mixed)

    def cursor(self) -> RngCursor:
        return RngCursor(self.state)

    def below(self, bound: int) -> tuple["Rng", int]:
        cursor = self.cursor()
        value = cursor.below(bound)
        return cursor.seal(), value

    def chance(self, numerator: int, denominator: int) -> tuple["Rng", bool]:
        cursor = self.cursor()
        value = cursor.chance(numerator, denominator)
        return cursor.seal(), value

    def choice(self, options: Sequence[T]) -> tuple["Rng", T]:
        cursor = self.cursor()
        value = cursor.choice(options)
        return cursor.seal(), value

    def split(self) -> tuple["Rng", "Rng"]:
        """Two independent streams from one. Useful for parallel rollouts."""
        cursor = self.cursor()
        left = cursor.bits()
        right = cursor.bits()
        return Rng(left), Rng(right)
