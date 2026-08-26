"""The unit of team building: a fully specified Pokemon, and its compiled form.

``PokemonSet`` is what a team builder produces and what a save file stores --
species, ability, item, four moves, Stat Alignment, SP spread. It is frozen and
hashable so it can be a dict key, cached on, and compared cheaply.

``BattlePokemon`` is the compiled form: the set with its dex entries resolved and
its stats already computed. Compilation happens once per battle, never per turn.
Everything on it is constant for the battle's duration, which is what lets a
``BattleState`` clone share it by reference instead of copying it (see
docs/DESIGN.md §1a).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from pkcm.data.dex import Dex, Move, Species, Stat
from pkcm.engine.stats import NEUTRAL_NATURE, Nature, StatTuple, compute_stats, get_nature

#: Champions computes fully-boosted PP as ``(pp / 5 + 1) * 4``, not the series'
#: ``pp * 8/5`` (mods/champions/scripts.ts, calculatePP). Combined with the base
#: PP cap of 20 the dex applies, Thunderbolt has 16 PP here where the mainline
#: games give it 24.
PP_STEP = 5
PP_BONUS = 1
PP_SCALE = 4

MAX_MOVES = 4

NO_SP: StatTuple = (0, 0, 0, 0, 0, 0)


def max_pp(base_pp: int) -> int:
    """Fully boosted PP, by Champions' formula."""
    if base_pp <= 1:
        return base_pp
    return (base_pp // PP_STEP + PP_BONUS) * PP_SCALE


@dataclass(frozen=True, slots=True)
class PokemonSet:
    """A team slot, as a team builder would write it."""

    species: str
    ability: str
    moves: tuple[str, ...]
    nature: str = NEUTRAL_NATURE.id
    sp: StatTuple = NO_SP
    item: str | None = None
    nickname: str | None = None

    def __post_init__(self) -> None:
        # Normalizing here keeps every downstream comparison and dict lookup honest.
        object.__setattr__(self, "moves", tuple(self.moves))
        object.__setattr__(self, "sp", tuple(self.sp))  # type: ignore[arg-type]

    def replace(self, **changes) -> "PokemonSet":
        from dataclasses import replace as _replace

        return _replace(self, **changes)


#: A registered team. Champions registers six and brings a subset into battle.
Team = tuple[PokemonSet, ...]


@dataclass(frozen=True, slots=True)
class BattlePokemon:
    """A ``PokemonSet`` with its dex entries resolved and stats precomputed.

    Constant for the whole battle. Never mutated, never copied per turn.
    """

    set: PokemonSet
    species: Species
    nature: Nature
    stats: StatTuple
    moves: tuple[Move, ...]
    max_pp: tuple[int, ...]
    ability: str
    item: str | None

    @property
    def max_hp(self) -> int:
        return self.stats[Stat.HP]

    @property
    def name(self) -> str:
        return self.set.nickname or self.species.name

    @property
    def types(self) -> tuple[str, ...]:
        return self.species.types

    def move_index(self, move_id: str) -> int:
        for index, move in enumerate(self.moves):
            if move.id == move_id:
                return index
        raise KeyError(f"{self.name} does not carry {move_id!r}")


def compile_set(dex: Dex, pokemon_set: PokemonSet) -> BattlePokemon:
    """Resolve a set against the dex. Raises ``KeyError`` on unknown ids.

    This does *not* check legality -- see ``pkcm.engine.legality``. A set can be
    compilable and still illegal (over the SP budget, an unlearnable move).
    """
    species = dex.species[pokemon_set.species]
    nature = get_nature(pokemon_set.nature)
    moves = tuple(dex.moves[move_id] for move_id in pokemon_set.moves)
    return BattlePokemon(
        set=pokemon_set,
        species=species,
        nature=nature,
        stats=compute_stats(species.base_stats, pokemon_set.sp, nature),
        moves=moves,
        max_pp=tuple(max_pp(move.pp) for move in moves),
        ability=pokemon_set.ability,
        item=pokemon_set.item,
    )


def compile_team(dex: Dex, team: Sequence[PokemonSet]) -> tuple[BattlePokemon, ...]:
    return tuple(compile_set(dex, pokemon_set) for pokemon_set in team)
