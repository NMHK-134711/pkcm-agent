"""Champions' stat system: Stat Points, Stat Alignments, and the level-50 formula.

Champions dropped the series' EV/IV model. Every Pokemon is treated as having
31 IVs in every stat, and EVs are replaced by **Stat Points (SP)**: 66 per
Pokemon, at most 32 in any one stat.

Because play is locked to level 50, the classic formula collapses into something
much simpler -- one SP is exactly one point of stat:

    HP     = floor(((2*Base + 31) * 50) / 100) + 50 + 10 + SP  ==  Base + 75 + SP
    others = floor((floor(((2*Base + 31) * 50) / 100) + 5 + SP) * alignment)
                                                      ==  floor((Base + 20 + SP) * alignment)

The identity holds because ``2*Base + 31`` is always odd, so the inner floor is
exactly ``Base + 15``. We implement the collapsed form and refuse other levels
rather than guessing how SP would scale off-level -- Champions never asks.

Stat Alignments are the series' natures minus four of the five neutral ones:
20 boost/hinder pairs plus Serious, 21 in total.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from pkcm.data.dex import Stat

#: Champions locks competitive play to level 50; the SP identity assumes it.
LEVEL = 50

#: Stat Points available per Pokemon, and the per-stat ceiling.
SP_TOTAL = 66
SP_PER_STAT_CAP = 32

#: Showdown computes the nature step as ``trunc(trunc(stat * 110, 16) / 100)``.
#: Integer arithmetic rather than a float multiply, so the result is exact --
#: the 16-bit truncation only bites above stat 595, which Champions cannot reach
#: (max base 255 + 20 + 32 SP = 307).
_BOOST_PERCENT = 110
_HINDER_PERCENT = 90
_NEUTRAL_PERCENT = 100

_BOOST = _BOOST_PERCENT / 100
_HINDER = _HINDER_PERCENT / 100

StatTuple = tuple[int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class Nature:
    """A Stat Alignment: +10% to one stat, -10% to another. HP is never touched."""

    name: str
    boosted: Stat | None
    hindered: Stat | None

    @property
    def id(self) -> str:
        return self.name.lower()

    @property
    def is_neutral(self) -> bool:
        return self.boosted is None

    def multiplier(self, stat: Stat) -> float:
        """The ratio, for display and for tests. ``percent`` is what stats use."""
        return self.percent(stat) / 100

    def percent(self, stat: Stat) -> int:
        if stat is Stat.HP:
            return _NEUTRAL_PERCENT
        if stat is self.boosted:
            return _BOOST_PERCENT
        if stat is self.hindered:
            return _HINDER_PERCENT
        return _NEUTRAL_PERCENT


def _build_natures() -> dict[str, Nature]:
    # (boosted, hindered) -> name, in the series' canonical order.
    pairs: dict[tuple[Stat, Stat], str] = {
        (Stat.ATK, Stat.DEF): "Lonely",
        (Stat.ATK, Stat.SPA): "Adamant",
        (Stat.ATK, Stat.SPD): "Naughty",
        (Stat.ATK, Stat.SPE): "Brave",
        (Stat.DEF, Stat.ATK): "Bold",
        (Stat.DEF, Stat.SPA): "Impish",
        (Stat.DEF, Stat.SPD): "Lax",
        (Stat.DEF, Stat.SPE): "Relaxed",
        (Stat.SPA, Stat.ATK): "Modest",
        (Stat.SPA, Stat.DEF): "Mild",
        (Stat.SPA, Stat.SPD): "Rash",
        (Stat.SPA, Stat.SPE): "Quiet",
        (Stat.SPD, Stat.ATK): "Calm",
        (Stat.SPD, Stat.DEF): "Gentle",
        (Stat.SPD, Stat.SPA): "Careful",
        (Stat.SPD, Stat.SPE): "Sassy",
        (Stat.SPE, Stat.ATK): "Timid",
        (Stat.SPE, Stat.DEF): "Hasty",
        (Stat.SPE, Stat.SPA): "Jolly",
        (Stat.SPE, Stat.SPD): "Naive",
    }
    natures = {
        name.lower(): Nature(name, boosted, hindered)
        for (boosted, hindered), name in pairs.items()
    }
    natures["serious"] = Nature("Serious", None, None)
    return natures


#: The 21 legal Stat Alignments, keyed by lowercase id.
NATURES: dict[str, Nature] = _build_natures()

NEUTRAL_NATURE = NATURES["serious"]


def get_nature(nature: Nature | str) -> Nature:
    if isinstance(nature, Nature):
        return nature
    try:
        return NATURES[nature.lower()]
    except KeyError:
        raise KeyError(f"{nature!r} is not a legal Champions Stat Alignment") from None


def compute_stat(base: int, sp: int, nature: Nature, stat: Stat, level: int = LEVEL) -> int:
    """One stat's level-50 value from its base stat and SP investment."""
    if level != LEVEL:
        raise ValueError(f"Champions locks battles to level {LEVEL}; got {level}")
    if stat is Stat.HP:
        return base + 75 + sp
    return (base + 20 + sp) * nature.percent(stat) // 100


def compute_stats(
    base_stats: Sequence[int],
    sp: Sequence[int],
    nature: Nature | str = NEUTRAL_NATURE,
    level: int = LEVEL,
) -> StatTuple:
    """All six stats, in ``Stat`` order."""
    resolved = get_nature(nature)
    return tuple(  # type: ignore[return-value]
        compute_stat(base_stats[stat], sp[stat], resolved, stat, level) for stat in Stat
    )


def sp_errors(sp: Sequence[int]) -> list[str]:
    """Every way an SP spread breaks the rules. Empty list means legal."""
    errors = []
    if len(sp) != len(Stat):
        errors.append(f"expected {len(Stat)} SP values, got {len(sp)}")
        return errors
    for stat in Stat:
        value = sp[stat]
        if value < 0:
            errors.append(f"{stat.name} SP is negative ({value})")
        elif value > SP_PER_STAT_CAP:
            errors.append(f"{stat.name} SP {value} exceeds the per-stat cap of {SP_PER_STAT_CAP}")
    total = sum(sp)
    if total > SP_TOTAL:
        errors.append(f"total SP {total} exceeds the budget of {SP_TOTAL}")
    return errors


def is_legal_sp(sp: Sequence[int]) -> bool:
    return not sp_errors(sp)


def spread_from_stats(stat_names: Iterable[str], amounts: Iterable[int]) -> StatTuple:
    """Build an SP tuple from a sparse spec, e.g. ``("spe", "atk"), (32, 32)``."""
    spread = [0] * len(Stat)
    for name, amount in zip(stat_names, amounts, strict=True):
        spread[Stat[name.upper()]] = amount
    return tuple(spread)  # type: ignore[return-value]
