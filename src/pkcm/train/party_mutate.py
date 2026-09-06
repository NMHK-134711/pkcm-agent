"""Small legal changes to a party, for a local search over party space.

DESIGN.md 6B: the sequential-construction MDP is not worth building -- there
is no data on what a half-built team is worth -- so the search is local
mutation from seeds people already made good. One operator at a time, so a
generation's result can be attributed to the change that caused it.

    species    one slot, within the regulation, no duplicate base species
    move       one of the four, from that species' learnset
    item       one slot, Item Clause holds
    ability    one slot, from what the dex allows it
    spread     a template drawn from the ranker sets, or a nudge inside one
    nature     one of the twenty-one

Everything hands back through ``team_errors``. An operator that cannot find a
legal change returns the team unchanged rather than an illegal one, and the
caller sees that nothing moved.

**On spreads.** 136 million spreads satisfy "sums to 66, at most 32 a stat",
and enumerating them is not a search, it is a random walk. The 276 ranker
sets say what people actually use: 91 of them are two stats at 32 with the
remaining 2 dumped in a third, 30 more are two at 32 and a 1/1, and 166 of
276 invest in exactly three stats. So the templates are those shapes, filled
in every assignment, and the fine-tuning is a few points moved between two
stats -- which is the "template plus adjustment" the design asked for,
measured rather than guessed.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from typing import Callable, Sequence

from pkcm.data.dex import Dex, Stat
from pkcm.engine.legality import (
    base_species_of,
    champions_items,
    holdable_items,
    mega_stone_for,
    ranker_slots,
    registrable_abilities,
    team_errors,
    usable_moves,
)
from pkcm.engine.pokemon import PokemonSet
from pkcm.engine.rng import RngCursor
from pkcm.engine.stats import NATURES, SP_PER_STAT_CAP, SP_TOTAL, StatTuple

Team = tuple[PokemonSet, ...]

#: How far a nudge may move points between two stats.
NUDGE_STEPS = (1, 2, 4, 8)


@lru_cache(maxsize=1)
def spread_templates() -> tuple[StatTuple, ...]:
    """Every spread the ranker sets use, and every relabelling of its shape.

    The shape is what generalises -- "two maxed and two spare" is a decision
    about a role, not about which stats a particular Pokemon has -- so each
    observed shape is expanded over the assignments of stats to it. The result
    is a few thousand spreads rather than a hundred and thirty-six million.
    """
    from itertools import permutations

    shapes = {tuple(sorted((v for v in one.sp if v), reverse=True))
              for one in ranker_slots()}
    out: set[StatTuple] = set()
    for shape in shapes:
        if sum(shape) != SP_TOTAL or any(v > SP_PER_STAT_CAP for v in shape):
            continue
        for stats in permutations(range(len(Stat)), len(shape)):
            spread = [0] * len(Stat)
            for stat, amount in zip(stats, shape):
                spread[stat] = amount
            out.add(tuple(spread))  # type: ignore[arg-type]
    return tuple(sorted(out))


def _species_pool(dex: Dex, regulation) -> list[str]:
    """Species the regulation allows, megas excluded -- a Mega is reached by
    holding the stone, not by registering the forme."""
    return sorted(one for one in regulation.legal_species
                  if not dex.species[one].is_mega)


def _slot(cursor: RngCursor, team: Team) -> int:
    return cursor.between(0, len(team) - 1)


def _swap(team: Team, index: int, **changes) -> Team:
    listed = list(team)
    listed[index] = replace(listed[index], **changes)
    return tuple(listed)


def _first_legal(dex: Dex, regulation, team: Team, battle_format: str,
                 candidates: Sequence[Team]) -> Team | None:
    for candidate in candidates:
        if not team_errors(dex, regulation, candidate, battle_format):
            return candidate
    return None


# --------------------------------------------------------------------------- #
# The operators. Each takes (dex, regulation, team, cursor, battle_format) and
# returns a team -- the same one if it could not find a legal change.
# --------------------------------------------------------------------------- #


def mutate_species(dex, regulation, team, cursor, battle_format="singles") -> Team:
    """Replace one Pokemon, keeping nothing of the old one.

    A new species cannot keep the old one's moves, and its item may not suit
    it either, so this draws a whole set: the ranker pool's own set for that
    species when there is one, since that is a set a person built.
    """
    index = _slot(cursor, team)
    taken = {base_species_of(dex, one.species)
             for i, one in enumerate(team) if i != index}
    pool = [one for one in _species_pool(dex, regulation)
            if base_species_of(dex, one) not in taken]
    if not pool:
        return team
    by_species: dict[str, list[PokemonSet]] = {}
    for one in ranker_slots():
        by_species.setdefault(one.species, []).append(one)

    tries = []
    for _ in range(12):
        species = pool[cursor.between(0, len(pool) - 1)]
        built = by_species.get(species)
        if built:
            fresh = built[cursor.between(0, len(built) - 1)]
        else:
            moves = usable_moves(dex, species)
            if len(moves) < 4:
                continue
            picked = tuple(moves[i] for i in
                           cursor.shuffled(list(range(len(moves))))[:4])
            abilities = registrable_abilities(dex.species[species]) or ("__none__",)
            fresh = PokemonSet(
                species=species, ability=abilities[cursor.between(0, len(abilities) - 1)],
                moves=picked, item=None,
                nature=sorted(NATURES)[cursor.between(0, len(NATURES) - 1)],
                sp=spread_templates()[cursor.between(0, len(spread_templates()) - 1)])
        tries.append(_swap(team, index, **{
            "species": fresh.species, "ability": fresh.ability,
            "moves": fresh.moves, "item": fresh.item,
            "nature": fresh.nature, "sp": fresh.sp}))
    return _first_legal(dex, regulation, team, battle_format, tries) or team


def mutate_move(dex, regulation, team, cursor, battle_format="singles") -> Team:
    """Swap one of a Pokemon's four moves for another it can learn."""
    index = _slot(cursor, team)
    one = team[index]
    pool = [m for m in usable_moves(dex, one.species) if m not in one.moves]
    if not pool:
        return team
    tries = []
    for _ in range(12):
        slot = cursor.between(0, len(one.moves) - 1)
        moves = list(one.moves)
        moves[slot] = pool[cursor.between(0, len(pool) - 1)]
        tries.append(_swap(team, index, moves=tuple(moves)))
    return _first_legal(dex, regulation, team, battle_format, tries) or team


def mutate_item(dex, regulation, team, cursor, battle_format="singles") -> Team:
    """Give one Pokemon a different item, Item Clause permitting.

    A Mega Stone is not drawn here. It changes what the Pokemon *is* rather
    than what it holds, and the team already carries at most one Mega; species
    mutation is where that decision belongs.
    """
    index = _slot(cursor, team)
    held = {one.item for i, one in enumerate(team) if i != index and one.item}
    stones = {mega_stone_for(dex, regulation, one.species) for one in team}
    pool = sorted((champions_items() & holdable_items(dex))
                  - held - {one for one in stones if one})
    if not pool:
        return team
    tries = [_swap(team, index, item=pool[cursor.between(0, len(pool) - 1)])
             for _ in range(12)]
    return _first_legal(dex, regulation, team, battle_format, tries) or team


def mutate_ability(dex, regulation, team, cursor, battle_format="singles") -> Team:
    index = _slot(cursor, team)
    one = team[index]
    pool = [a for a in registrable_abilities(dex.species[one.species])
            if a != one.ability]
    if not pool:
        return team
    tries = [_swap(team, index, ability=pool[cursor.between(0, len(pool) - 1)])
             for _ in range(len(pool))]
    return _first_legal(dex, regulation, team, battle_format, tries) or team


def mutate_spread(dex, regulation, team, cursor, battle_format="singles") -> Team:
    """A different template for one Pokemon, drawn from the shapes people use."""
    index = _slot(cursor, team)
    templates = spread_templates()
    tries = []
    for _ in range(12):
        fresh = templates[cursor.between(0, len(templates) - 1)]
        if fresh != team[index].sp:
            tries.append(_swap(team, index, sp=fresh))
    return _first_legal(dex, regulation, team, battle_format, tries) or team


def nudge_spread(dex, regulation, team, cursor, battle_format="singles") -> Team:
    """Move a few points from one stat to another, inside the same template.

    The fine adjustment the templates cannot express: a survival benchmark is
    usually a handful of points, not a stat.
    """
    index = _slot(cursor, team)
    spread = list(team[index].sp)
    tries = []
    for _ in range(12):
        give = cursor.between(0, len(spread) - 1)
        take = cursor.between(0, len(spread) - 1)
        step = NUDGE_STEPS[cursor.between(0, len(NUDGE_STEPS) - 1)]
        if give == take or spread[give] < step:
            continue
        if spread[take] + step > SP_PER_STAT_CAP:
            continue
        fresh = list(spread)
        fresh[give] -= step
        fresh[take] += step
        tries.append(_swap(team, index, sp=tuple(fresh)))
    return _first_legal(dex, regulation, team, battle_format, tries) or team


def mutate_nature(dex, regulation, team, cursor, battle_format="singles") -> Team:
    index = _slot(cursor, team)
    pool = [n for n in sorted(NATURES) if n != team[index].nature]
    tries = [_swap(team, index, nature=pool[cursor.between(0, len(pool) - 1)])
             for _ in range(12)]
    return _first_legal(dex, regulation, team, battle_format, tries) or team


Operator = Callable[..., Team]

#: The operators and how often to reach for each. Weighted towards the small
#: ones: a species swap discards four moves, an item and a spread at once, so
#: it is a jump rather than a step, and a local search made of jumps is a
#: random search.
OPERATORS: tuple[tuple[str, Operator, int], ...] = (
    ("move", mutate_move, 5),
    ("item", mutate_item, 4),
    ("spread", mutate_spread, 3),
    ("nudge", nudge_spread, 3),
    ("ability", mutate_ability, 2),
    ("nature", mutate_nature, 2),
    ("species", mutate_species, 3),
)


def mutate(dex: Dex, regulation, team: Team, cursor: RngCursor,
           battle_format: str = "singles") -> tuple[str, Team]:
    """One weighted operator, applied once. Returns its name and the result."""
    total = sum(weight for _, _, weight in OPERATORS)
    roll = cursor.between(0, total - 1)
    for name, operator, weight in OPERATORS:
        if roll < weight:
            return name, operator(dex, regulation, team, cursor, battle_format)
        roll -= weight
    name, operator, _ = OPERATORS[-1]
    return name, operator(dex, regulation, team, cursor, battle_format)
