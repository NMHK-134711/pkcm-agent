"""Team legality under a Champions regulation, and random legal team generation.

The subtle rule is the **Species Clause**: it binds on the *base species*, not on
the forme. Goodra and Goodra-Hisui are one species for this purpose and cannot
share a team, and the same goes for the three Paldean Tauros breeds, Meowstic's
two genders, and Gourgeist's four sizes. Checking forme ids would silently let
all of those through.

Move legality is an approximation. Champions' own learnsets are not published, so
we accept the union of every generation's learnset from Showdown, restricted to
moves that are standard in the current generation (which drops Z-Moves, Max
Moves, and moves cut from the modern games). This is permissive: a set legal here
may be illegal in game. Tighten ``learnable_moves`` when real data appears.
"""

from __future__ import annotations

from dataclasses import dataclass


from pkcm.data.dex import Dex, Regulation, Stat
from pkcm.engine.pokemon import MAX_MOVES, PokemonSet, Team
from pkcm.engine.rng import RngCursor
from pkcm.engine.scope import is_supported
from pkcm.engine.stats import NATURES, SP_PER_STAT_CAP, SP_TOTAL, StatTuple, sp_errors


#: Moves that exist upstream but depend on a mechanic Champions does not have.
#: Terastallization is absent from the M-B ruleset, so these can never be legal.
ABSENT_MECHANIC_MOVES = frozenset({"terablast", "terastarstorm"})


def champions_items() -> frozenset[str]:
    """Every item id Champions actually has. Cached; the file never changes."""
    global _CHAMPIONS_ITEMS
    if _CHAMPIONS_ITEMS is None:
        from pkcm.engine.items import champions_items as roster

        _CHAMPIONS_ITEMS = frozenset(roster())
    return _CHAMPIONS_ITEMS


_CHAMPIONS_ITEMS: frozenset[str] | None = None


def clause_violation(move) -> str | None:
    """Champions' standard ruleset bans three whole categories of move.

    From ``mods/champions/rulesets.ts``: the ``standard`` ruleset adds Sleep
    Moves Clause, OHKO Clause and Evasion Clause on top of Species and Item
    Clause. Checking the move's data rather than a hand-written ban list means
    new moves are covered automatically.
    """
    if move.raw.get("status") == "slp":
        return "sleep moves clause"
    if move.raw.get("ohko"):
        return "OHKO clause"
    boosts = move.raw.get("boosts") or {}
    if boosts.get("evasion", 0) > 0:
        return "evasion clause"
    secondary = move.raw.get("secondary") or {}
    if (secondary.get("self") or {}).get("boosts", {}).get("evasion", 0) > 0:
        return "evasion clause"
    return None


def _learnset_sources(dex: Dex, species_id: str) -> list[str]:
    """Which learnset entries make up this forme's pool, in priority order.

    Two different inheritance shapes hide in Showdown's data:

    * A forme with ``changesFrom`` (Rotom's appliances) stores *only* its
      signature move and inherits everything else. Rotom-Heat's own entry has a
      single move in it -- read it alone and the Pokemon cannot do anything.
    * A regional forme (Alolan Raichu, Hisuian Arcanine) stores a complete pool
      of its own and must *not* inherit, or it would gain moves its Kantonian
      counterpart learns and it does not.
    * A cosmetic forme (Gourgeist's sizes) has no entry at all and falls back.
    """
    sources: list[str] = []
    queue = [species_id]
    while queue:
        current = queue.pop(0)
        if current in sources:
            continue
        sources.append(current)
        species = dex.species.get(current)
        if species is None:
            continue
        entry = dex.learnsets.get(current)
        if species.changes_from:
            queue.append(species.changes_from)
        elif not (entry and entry.get("learnset")) and species.base_species != current:
            queue.append(species.base_species)
    return sources


def _compute_learnable(dex: Dex, species_id: str) -> frozenset[str]:
    move_ids: set[str] = set()
    for source in _learnset_sources(dex, species_id):
        entry = dex.learnsets.get(source)
        if entry:
            move_ids.update(entry.get("learnset", {}))
    return frozenset(
        move_id
        for move_id in move_ids
        if move_id in dex.moves
        and dex.moves[move_id].raw.get("isNonstandard") is None
        and move_id not in ABSENT_MECHANIC_MOVES
        and clause_violation(dex.moves[move_id]) is None
    )


def learnable_moves(dex: Dex, species_id: str) -> frozenset[str]:
    """Every standard move ``species_id`` can learn, across all generations.

    Memoized on the dex instance rather than in a module-level cache, so the
    entries die with the dex instead of outliving it.
    """
    try:
        cache = dex._learnable_moves  # type: ignore[attr-defined]
    except AttributeError:
        cache = dex._learnable_moves = {}  # type: ignore[attr-defined]
    result = cache.get(species_id)
    if result is None:
        result = cache[species_id] = _compute_learnable(dex, species_id)
    return result


def base_species_of(dex: Dex, species_id: str) -> str:
    """The id the Species Clause actually binds on."""
    return dex.species[species_id].base_species


def set_errors(dex: Dex, regulation: Regulation, pokemon_set: PokemonSet) -> list[str]:
    """Everything wrong with one team slot. Empty list means legal."""
    errors: list[str] = []
    label = pokemon_set.species

    species = dex.species.get(pokemon_set.species)
    if species is None:
        return [f"{label}: unknown species"]
    if pokemon_set.species not in regulation.legal_species:
        if pokemon_set.species in regulation.legal_megas:
            errors.append(
                f"{label}: Mega formes are reached in battle, not registered on a team"
            )
        else:
            errors.append(f"{label}: not eligible in Regulation {regulation.name}")

    if pokemon_set.ability not in species.abilities:
        errors.append(
            f"{label}: ability {pokemon_set.ability!r} is not one of {list(species.abilities)}"
        )

    if pokemon_set.item is not None:
        if pokemon_set.item not in dex.items:
            errors.append(f"{label}: unknown item {pokemon_set.item!r}")
        elif pokemon_set.item not in champions_items():
            errors.append(
                f"{label}: {dex.items[pokemon_set.item].name} does not exist in Champions"
            )

    if not 1 <= len(pokemon_set.moves) <= MAX_MOVES:
        errors.append(f"{label}: carries {len(pokemon_set.moves)} moves, must be 1-{MAX_MOVES}")
    if len(set(pokemon_set.moves)) != len(pokemon_set.moves):
        errors.append(f"{label}: duplicate moves")

    legal_moves = learnable_moves(dex, pokemon_set.species)
    for move_id in pokemon_set.moves:
        if move_id not in dex.moves:
            errors.append(f"{label}: unknown move {move_id!r}")
            continue
        move = dex.moves[move_id]
        if move.raw.get("isNonstandard") is not None:
            errors.append(f"{label}: {move.name} does not exist in Champions")
        elif (clause := clause_violation(move)) is not None:
            errors.append(f"{label}: {move.name} is banned by the {clause}")
        elif move_id not in legal_moves:
            errors.append(f"{label}: cannot learn {move_id!r}")

    if pokemon_set.nature not in NATURES:
        errors.append(f"{label}: {pokemon_set.nature!r} is not a legal Stat Alignment")

    errors.extend(f"{label}: {error}" for error in sp_errors(pokemon_set.sp))
    return errors


def team_errors(
    dex: Dex,
    regulation: Regulation,
    team: Team,
    battle_format: str = "singles",
) -> list[str]:
    """Everything wrong with a whole team. Empty list means legal."""
    errors: list[str] = []

    registered, _ = regulation.bring_select(battle_format)
    if len(team) != registered:
        errors.append(f"team has {len(team)} Pokemon, Regulation {regulation.name} registers {registered}")

    for pokemon_set in team:
        errors.extend(set_errors(dex, regulation, pokemon_set))

    # Species Clause, on the base species -- regional formes collide.
    seen_species: dict[str, str] = {}
    for pokemon_set in team:
        if pokemon_set.species not in dex.species:
            continue
        base = base_species_of(dex, pokemon_set.species)
        if base in seen_species:
            first = seen_species[base]
            if first == pokemon_set.species:
                errors.append(f"species clause: {pokemon_set.species} appears twice")
            else:
                errors.append(
                    f"species clause: {first} and {pokemon_set.species} are both {base}"
                )
        else:
            seen_species[base] = pokemon_set.species

    # Item Clause. Holding nothing is not an item, so it may repeat.
    seen_items: set[str] = set()
    for pokemon_set in team:
        if pokemon_set.item is None:
            continue
        if pokemon_set.item in seen_items:
            errors.append(f"item clause: {pokemon_set.item} is held more than once")
        seen_items.add(pokemon_set.item)

    return errors


def is_legal_team(dex: Dex, regulation: Regulation, team: Team, battle_format: str = "singles") -> bool:
    return not team_errors(dex, regulation, team, battle_format)


# --------------------------------------------------------------------------- #
# Random legal teams. These exist so self-play has something to fight with
# before a real team builder exists; they are not meant to be good teams.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RandomTeamOptions:
    #: Held items are implemented, so random teams use them.
    with_items: bool = True
    #: How often a Pokemon that *could* hold its Mega Stone actually does.
    #: Not always: one Mega per battle means a second stone is a wasted slot,
    #: and a policy should see both kinds of team.
    mega_stone_chance: int = 40
    #: Without a damaging move a Pokemon cannot win, and battles never end.
    require_damaging_move: bool = True
    moves_per_pokemon: int = MAX_MOVES
    #: Draw only from moves the engine executes faithfully. Off by default is
    #: not an option worth having: self-play on moves that resolve as no-ops
    #: teaches the policy that those moves are free, which is worse than useless.
    only_supported_moves: bool = True


def random_sp(cursor: RngCursor) -> StatTuple:
    """A legal SP spread, weighted toward the concentrated spreads people use."""
    spread = [0] * len(Stat)
    budget = SP_TOTAL
    for stat in cursor.shuffled(list(Stat)):
        if budget <= 0:
            break
        # Bias toward dumping the cap into the first stats we touch.
        amount = min(SP_PER_STAT_CAP, budget, cursor.between(0, SP_PER_STAT_CAP))
        spread[stat] = amount
        budget -= amount
    return tuple(spread)  # type: ignore[return-value]


def usable_moves(
    dex: Dex,
    species_id: str,
    options: "RandomTeamOptions" = None,  # type: ignore[assignment]
) -> list[str]:
    """Learnable moves, narrowed to what the generator may draw from."""
    options = options or RandomTeamOptions()
    try:
        cache = dex._usable_moves  # type: ignore[attr-defined]
    except AttributeError:
        cache = dex._usable_moves = {}  # type: ignore[attr-defined]

    key = (species_id, options)
    pool = cache.get(key)
    if pool is None:
        pool = sorted(learnable_moves(dex, species_id))
        if options.only_supported_moves:
            pool = [move_id for move_id in pool if is_supported(dex.moves[move_id])]
        if options.require_damaging_move and not any(dex.moves[m].base_power > 0 for m in pool):
            pool = []
        cache[key] = pool
    return pool


def random_set(
    dex: Dex,
    species_id: str,
    cursor: RngCursor,
    options: RandomTeamOptions = RandomTeamOptions(),
) -> PokemonSet:
    species = dex.species[species_id]
    pool = usable_moves(dex, species_id, options)
    if not pool:
        raise ValueError(f"{species_id} has no moves this engine can execute")

    count = min(options.moves_per_pokemon, len(pool))
    moves = cursor.sample(pool, count)

    if options.require_damaging_move and not any(dex.moves[m].base_power > 0 for m in moves):
        damaging = [m for m in pool if dex.moves[m].base_power > 0]
        if damaging:
            moves[cursor.below(len(moves))] = cursor.choice(damaging)

    return PokemonSet(
        species=species_id,
        ability=cursor.choice(species.abilities),
        moves=tuple(moves),
        nature=cursor.choice(sorted(NATURES)),
        sp=random_sp(cursor),
        item=None,
        gender=cursor.choice(("M", "F")) if species.gender is None else species.gender,
    )


def random_team(
    dex: Dex,
    regulation: Regulation,
    cursor: RngCursor,
    battle_format: str = "singles",
    options: RandomTeamOptions = RandomTeamOptions(),
) -> Team:
    """A random team that satisfies every clause, including Species Clause."""
    registered, _ = regulation.bring_select(battle_format)

    # Draw base species first, so the Species Clause holds by construction
    # rather than by rejection sampling.
    by_base: dict[str, list[str]] = {}
    for species_id in regulation.legal_species:
        if not usable_moves(dex, species_id, options):
            # Ditto is legal and learns only Transform: nothing M0 can execute.
            continue
        by_base.setdefault(base_species_of(dex, species_id), []).append(species_id)

    if len(by_base) < registered:
        raise ValueError(
            f"only {len(by_base)} base species have moves this engine supports, "
            f"need {registered}"
        )

    chosen_bases = cursor.sample(sorted(by_base), registered)
    team = [
        random_set(dex, cursor.choice(sorted(by_base[base])), cursor, options)
        for base in chosen_bases
    ]

    if options.with_items:
        team = _deal_items(dex, regulation, team, cursor, options)

    return tuple(team)


def _deal_items(dex, regulation, team, cursor: RngCursor, options) -> list[PokemonSet]:
    """One item each, Item Clause respected, Mega Stones only where usable."""
    pool = [item for item in sorted(holdable_items(dex))]
    taken: set[str] = set()
    dealt: list[PokemonSet] = []

    for pokemon in team:
        stone = mega_stone_for(dex, regulation, pokemon.species)
        if (stone is not None and stone not in taken
                and cursor.chance(options.mega_stone_chance, 100)):
            taken.add(stone)
            dealt.append(pokemon.replace(item=stone))
            continue
        choices = [item for item in pool if item not in taken]
        chosen = cursor.choice(choices)
        taken.add(chosen)
        dealt.append(pokemon.replace(item=chosen))
    return dealt


def mega_stone_for(dex: Dex, regulation: Regulation, species_id: str) -> str | None:
    """The legal Mega Stone this species can actually use, if there is one."""
    for item_id, mega_ids in regulation.legal_mega_stones.items():
        if dex.mega_evolution(species_id, item_id) in mega_ids:
            return item_id
    return None


def holdable_items(dex: Dex) -> frozenset[str]:
    """Champions items worth giving a random team: everything but Mega Stones."""
    global _HOLDABLE
    if _HOLDABLE is None:
        # Stones are dealt separately -- they are only useful to their species.
        _HOLDABLE = frozenset(
            item_id for item_id in champions_items()
            if not dex.items[item_id].mega_stone
        )
    return _HOLDABLE


_HOLDABLE: frozenset[str] | None = None
