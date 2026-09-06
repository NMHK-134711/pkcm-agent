"""Spreads aimed at a threat, computed rather than searched for.

The thing people actually do when they build: put in exactly enough Speed to
move before a Pokemon they expect to meet, exactly enough bulk to survive two
of its hits, and everything left over into offence. Champions makes both
exact -- one SP is one stat point -- so these are arithmetic, and a random
walk over spreads is the wrong tool for them twice over.

**The search cannot find them.** The templates offer every Speed value from 0
to 32, so reachability is not the problem; the problem is that one of the
thirty-three is right and the fitness cannot tell which. Garchomp is in 25 of
the 46 field parties, the most common threat there is, and with three of six
brought it appears in roughly 27% of games. A Speed tier that swings that
matchup by ten points moves the overall win rate by 2.7, which needs about
1300 games a candidate to separate -- and that is the *easiest* threat in the
field. A generation of thirty candidates cannot pay that, so the signal a
benchmark produces is below the noise the search is reading.

**And people do not find them by searching either.** They compute them. So
this proposes the computed spread as a candidate and lets the games judge it,
which is the same division of labour: arithmetic proposes, measurement
disposes. A benchmark built on a wrong guess about the threat's set is simply
a candidate that loses.

The damage arithmetic here is deliberately plain -- STAB and the type chart,
no items, abilities, weather or screens. It decides which candidates are
worth playing, not what is true, and every one of them is then measured in
real games where all of that applies.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from functools import lru_cache

from pkcm.data.dex import Dex, Stat
from pkcm.engine.legality import ranker_parties, team_errors
from pkcm.engine.moves import DAMAGE_ROLL_HIGH, DAMAGE_ROLL_LOW, damage_formula
from pkcm.engine.pokemon import PokemonSet
from pkcm.engine.rng import RngCursor
from pkcm.engine.stats import (
    NATURES,
    SP_PER_STAT_CAP,
    SP_TOTAL,
    compute_stat,
    get_nature,
)

Team = tuple[PokemonSet, ...]

#: What "survives" is asked at. Both are proposed and the games choose: the
#: high roll is the guarantee people build to, the low roll is the cheaper bet
#: that it will not come to that.
ROLLS = {"guaranteed": DAMAGE_ROLL_HIGH, "optimistic": DAMAGE_ROLL_LOW}


@lru_cache(maxsize=4)
def threats(parties_path: str | None = None) -> tuple[tuple[PokemonSet, int], ...]:
    """Every set in the field with how many parties carry that species.

    Weighted by how often it will actually be across the table, which is what
    decides whether aiming at it is worth the points.
    """
    field = ranker_parties(parties_path)
    carried: Counter[str] = Counter()
    for party in field:
        for species in {one.species for one in party.team}:
            carried[species] += 1
    seen: dict[tuple, PokemonSet] = {}
    for party in field:
        for one in party.team:
            seen.setdefault((one.species, one.moves, one.item, one.sp, one.nature), one)
    return tuple((one, carried[one.species]) for one in seen.values())


def _pick_threat(cursor: RngCursor, pool) -> PokemonSet:
    """Draw a threat in proportion to how many parties carry its species."""
    total = sum(weight for _, weight in pool)
    roll = cursor.between(0, max(0, total - 1))
    for one, weight in pool:
        if roll < weight:
            return one
        roll -= weight
    return pool[-1][0]


def speed_to_outrun(dex: Dex, ours: PokemonSet, theirs: PokemonSet) -> int | None:
    """The least Speed SP that moves us before ``theirs``, or ``None``.

    ``None`` covers both "already faster with nothing" -- which needs no
    points and so is not a benchmark -- and "cannot get there with all
    thirty-two", which is a spread not worth proposing.
    """
    base = dex.species[ours.species].base_stats[Stat.SPE]
    nature = get_nature(ours.nature)
    other = compute_stat(dex.species[theirs.species].base_stats[Stat.SPE],
                         theirs.sp[Stat.SPE], get_nature(theirs.nature), Stat.SPE)
    for sp in range(SP_PER_STAT_CAP + 1):
        if compute_stat(base, sp, nature, Stat.SPE) > other:
            return None if sp == 0 else sp
    return None


def _hit(dex: Dex, attacker: PokemonSet, defender: PokemonSet, move_id: str,
         defence_sp: int, hp_sp: int, roll: int) -> tuple[int, int]:
    """One hit's damage and the defender's HP, on the plain arithmetic."""
    move = dex.moves[move_id]
    attack_stat = Stat.ATK if move.category == "Physical" else Stat.SPA
    defence_stat = Stat.DEF if move.category == "Physical" else Stat.SPD
    attack = compute_stat(dex.species[attacker.species].base_stats[attack_stat],
                          attacker.sp[attack_stat], get_nature(attacker.nature),
                          attack_stat)
    defence = compute_stat(dex.species[defender.species].base_stats[defence_stat],
                           defence_sp, get_nature(defender.nature), defence_stat)
    effectiveness = dex.type_chart.multiplier(
        move.type, dex.species[defender.species].types)
    damage = damage_formula(
        power=move.base_power, attack=attack, defense=defence, roll=roll,
        stab=move.type in dex.species[attacker.species].types,
        effectiveness=effectiveness)
    hp = compute_stat(dex.species[defender.species].base_stats[Stat.HP],
                      hp_sp, get_nature(defender.nature), Stat.HP)
    return damage, hp


def worst_move(dex: Dex, attacker: PokemonSet, defender: PokemonSet) -> str | None:
    """Whichever of the threat's four hurts us most, at a neutral spread."""
    best, worst = 0, None
    for move_id in attacker.moves:
        move = dex.moves.get(move_id)
        if move is None or move.category == "Status" or not move.base_power:
            continue
        damage, _ = _hit(dex, attacker, defender, move_id, 0, 0, DAMAGE_ROLL_HIGH)
        if damage > best:
            best, worst = damage, move_id
    return worst


def bulk_to_survive(dex: Dex, ours: PokemonSet, theirs: PokemonSet, move_id: str,
                    hits: int, roll: int, spare: int) -> tuple[int, int] | None:
    """The cheapest HP and defence investment that lives through ``hits``.

    Cheapest by total points, searched over the split rather than over one
    stat: HP raises what every attack has to get through and defence only the
    matching side, so which is worth more depends on the number and there is
    no rule of thumb that survives being written down.
    """
    move = dex.moves.get(move_id)
    if move is None or move.category == "Status" or not move.base_power:
        return None
    best = None
    for total in range(0, spare + 1):
        for hp_sp in range(0, min(total, SP_PER_STAT_CAP) + 1):
            defence_sp = total - hp_sp
            if defence_sp > SP_PER_STAT_CAP:
                continue
            damage, hp = _hit(dex, theirs, ours, move_id,
                              defence_sp, hp_sp, roll)
            if damage * hits < hp:
                best = (hp_sp, defence_sp)
                break
        if best is not None:
            return best
    return None


def _rebalance(spread: list[int], keep: set[int], into: list[int]) -> tuple | None:
    """Spend the leftover points on ``into``, respecting the total and the cap."""
    used = sum(spread[stat] for stat in keep)
    if used > SP_TOTAL:
        return None
    left = SP_TOTAL - used
    fresh = [0] * len(spread)
    for stat in keep:
        fresh[stat] = spread[stat]
    for stat in into:
        if left <= 0:
            break
        give = min(SP_PER_STAT_CAP - fresh[stat], left)
        fresh[stat] += give
        left -= give
    if left:
        for stat in range(len(fresh)):
            if left <= 0:
                break
            give = min(SP_PER_STAT_CAP - fresh[stat], left)
            fresh[stat] += give
            left -= give
    return tuple(fresh) if sum(fresh) == SP_TOTAL else None


def _offence_order(dex: Dex, one: PokemonSet) -> list[int]:
    """Which stats the leftover points should go to, best first."""
    base = dex.species[one.species].base_stats
    physical = sum(1 for m in one.moves
                   if (mv := dex.moves.get(m)) and mv.category == "Physical")
    special = sum(1 for m in one.moves
                  if (mv := dex.moves.get(m)) and mv.category == "Special")
    main = Stat.ATK if physical >= special else Stat.SPA
    other = Stat.SPD if base[Stat.SPD] >= base[Stat.DEF] else Stat.DEF
    return [main, Stat.SPE, Stat.HP, other]


# --------------------------------------------------------------------------- #
# The operators, in the same shape as party_mutate's
# --------------------------------------------------------------------------- #


def benchmark_speed(dex: Dex, regulation, team: Team, cursor: RngCursor,
                    battle_format: str = "singles") -> Team:
    """Put in exactly enough Speed to move before a threat, and no more.

    "No more" is the half a random walk never learns. Thirty-two into Speed
    when seventeen would do is fifteen points not spent on anything, and the
    difference is far too small for a win rate over a field to see.
    """
    pool = threats()
    index = cursor.between(0, len(team) - 1)
    ours = team[index]
    tries = []
    for _ in range(16):
        threat = _pick_threat(cursor, pool)
        needed = speed_to_outrun(dex, ours, threat)
        if needed is None:
            continue
        spread = list(ours.sp)
        spread[Stat.SPE] = needed
        order = [s for s in _offence_order(dex, ours) if s is not Stat.SPE]
        fresh = _rebalance(spread, {Stat.SPE}, order)
        if fresh is None or fresh == ours.sp:
            continue
        listed = list(team)
        listed[index] = replace(ours, sp=fresh)
        tries.append(tuple(listed))
    for candidate in tries:
        if not team_errors(dex, regulation, candidate, battle_format):
            return candidate
    return team


def benchmark_bulk(dex: Dex, regulation, team: Team, cursor: RngCursor,
                   battle_format: str = "singles") -> Team:
    """Put in exactly enough bulk to live through a threat's best two hits.

    Both readings of "enough" are proposed and the games pick between them:
    the high roll is the guarantee people build to, the low roll is the
    cheaper bet that it does not come to that. Which is right is a question
    about risk that nobody has to answer in the abstract here.
    """
    pool = threats()
    index = cursor.between(0, len(team) - 1)
    ours = team[index]
    tries = []
    for _ in range(16):
        threat = _pick_threat(cursor, pool)
        move_id = worst_move(dex, threat, ours)
        if move_id is None:
            continue
        kind = "guaranteed" if cursor.between(0, 1) else "optimistic"
        hits = 2
        found = bulk_to_survive(dex, ours, threat, move_id, hits,
                                ROLLS[kind], SP_TOTAL)
        if found is None:
            continue
        hp_sp, defence_sp = found
        move = dex.moves[move_id]
        defence_stat = Stat.DEF if move.category == "Physical" else Stat.SPD
        spread = [0] * len(ours.sp)
        spread[Stat.HP] = hp_sp
        spread[defence_stat] = defence_sp
        order = [s for s in _offence_order(dex, ours)
                 if s not in (Stat.HP, defence_stat)]
        fresh = _rebalance(spread, {Stat.HP, defence_stat}, order)
        if fresh is None or fresh == ours.sp:
            continue
        listed = list(team)
        listed[index] = replace(ours, sp=fresh)
        tries.append(tuple(listed))
    for candidate in tries:
        if not team_errors(dex, regulation, candidate, battle_format):
            return candidate
    return team
