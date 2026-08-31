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

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


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


#: Abilities Champions has in its data and forbids by rule. hk confirmed Battle
#: Bond is the banned kind rather than the missing kind, which is why this is a
#: clause and not a hole in the dex: the ability exists, and no team may bring
#: it. Written out because a ban is a decision, not something derivable from the
#: ability's own data the way the move clauses are.
BANNED_ABILITIES = {
    "battlebond": "battle bond clause",
}


def ability_clause(ability_id: str) -> str | None:
    """Why this ability may not be registered, or ``None`` if it may."""
    return BANNED_ABILITIES.get(ability_id)


def registrable_abilities(species) -> tuple[str, ...]:
    """The species' abilities minus the ones the format forbids.

    Falls back to the full list if a ban would leave nothing -- a species whose
    every ability is banned would be unbuildable, and that is a mistake in the
    ban list rather than a team the game refuses to accept.
    """
    allowed = tuple(a for a in species.abilities if ability_clause(a) is None)
    return allowed or species.abilities


def clause_violation(move) -> str | None:
    """What the format refuses to let a team carry.

    ``mods/champions/rulesets.ts`` lists Sleep Moves Clause, OHKO Clause and
    Evasion Clause in its ``standard`` ruleset, and this used to enforce all
    three. **Two of them are not in the game.**

    hk confirmed Sleep Powder is usable, and the ladder archive agrees at a
    scale that is hard to argue with: of 113 ranker parties from 2400-2800,
    **eleven carry a sleep move** -- Sleep Powder ten times and Hypnosis once --
    and three carry a one-hit-KO move (Fissure, Horn Drill, Guillotine). Those
    are teams that were actually played.

    So the Showdown mod is describing a ruleset the game does not run, and
    docs/HANDOFF.md's note that "Hypnosis and Sing are in the table and cannot
    go on a team" was our inference from it rather than an observation.

    Evasion Clause stays: nothing in the archive uses an evasion move, which is
    no evidence either way, and removing a ban on no evidence is how a format
    quietly stops being the format.
    """
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


#: The Champions learnset table, built from the 포케챔스 dex by
#: ``scripts/build_champions_learnsets.py``. Empty if the file is absent, which
#: falls the whole thing back to the union below.
_CHAMPIONS_LEARNSETS: dict[str, frozenset[str]] | None = None


def champions_learnsets() -> dict[str, frozenset[str]]:
    """Species id -> what Champions actually teaches it.

    This is the table, not an approximation of it. Showdown's ``learnsets.json``
    is an all-generations record of the *main series*, and it is wrong in both
    directions here: it keeps TM moves Champions dropped, and misses egg and
    tutor moves Champions gives out.
    """
    global _CHAMPIONS_LEARNSETS
    if _CHAMPIONS_LEARNSETS is None:
        path = (Path(__file__).resolve().parents[3]
                / "data" / "champions" / "learnsets.json")
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            _CHAMPIONS_LEARNSETS = {
                species: frozenset(moves) for species, moves in raw.items()
            }
        else:
            _CHAMPIONS_LEARNSETS = {}
    return _CHAMPIONS_LEARNSETS


def _champions_entry(dex: Dex, species_id: str) -> frozenset[str] | None:
    """This species' row, or its base species' -- cosmetic formes share one.

    The dex splits Vivillon into twenty patterns; the game lists one. They all
    learn the same moves, so the base species' row is the right answer rather
    than a fallback.
    """
    table = champions_learnsets()
    if species_id in table:
        return table[species_id]
    base = dex.species[species_id].base_species
    return table.get(base)


def _compute_learnable(dex: Dex, species_id: str) -> frozenset[str]:
    taught = _champions_entry(dex, species_id)
    if taught is not None:
        # Still filtered: the table says what the game teaches, and the clauses
        # say what the format allows. Two different questions, and a move can
        # be taught and still banned (Hypnosis, Double Team).
        return frozenset(
            move_id for move_id in taught
            if move_id in dex.moves
            and move_id not in ABSENT_MECHANIC_MOVES
            and clause_violation(dex.moves[move_id]) is None
        )

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

    banned = ability_clause(pokemon_set.ability)
    if banned is not None:
        errors.append(f"{label}: ability {pokemon_set.ability!r} is banned by the {banned}")
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
        ability=cursor.choice(registrable_abilities(species)),
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


# --------------------------------------------------------------------------- #
# Teams built out of what people actually brought
# --------------------------------------------------------------------------- #

#: Where the imported ranker slots live. Written by ``scripts/import_parties.py``
#: from the pkmnchamps archive; committed, because the archive itself is not.
PARTIES_PATH = Path(__file__).resolve().parents[3] / "data" / "champions" / "parties_m_b.json"


@dataclass(frozen=True, slots=True)
class Party:
    """One imported ranker party, kept whole.

    ``ranker_slots`` below throws the team away and keeps the Pokemon, which is
    what training wants. A tournament between teams cannot use that: the whole
    question there is whether *this* six beats *that* six, and the team's own
    idea -- the sand setter and the sweeper that needs it -- is the thing being
    measured rather than the thing being averaged out.
    """

    title: str
    #: The author's ladder rating and placing, where the archive recorded them.
    rate: int | None
    rank: int | None
    team: Team


def _party_payload(path: str | None) -> list[dict]:
    target = Path(path) if path else PARTIES_PATH
    if not target.exists():
        raise FileNotFoundError(
            f"{target} is not there -- run scripts/import_parties.py")
    return json.loads(target.read_text(encoding="utf-8"))


def _party_set(entry: dict) -> PokemonSet:
    return PokemonSet(
        species=entry["species"], ability=entry["ability"],
        moves=tuple(entry["moves"]), item=entry.get("item"),
        nature=entry["nature"], sp=tuple(entry["sp"]),
        gender=entry.get("gender"))


@lru_cache(maxsize=4)
def ranker_parties(path: str | None = None) -> tuple[Party, ...]:
    """The imported parties as their authors built them, in file order."""
    return tuple(
        Party(title=party["title"], rate=party.get("rate"),
              rank=party.get("rank"),
              team=tuple(_party_set(entry) for entry in party["team"]))
        for party in _party_payload(path)
    )


@lru_cache(maxsize=4)
def ranker_slots(path: str | None = None) -> tuple[PokemonSet, ...]:
    """Every Pokemon from every imported ranker party, as a flat pool.

    Twenty parties is not enough to train on as twenty parties -- the same six
    coming round again teaches the network which battle it is looking at, which
    is the mistake ``search_value_weight`` is switched off for. A hundred and
    twenty *slots* recombine into far more teams than that, and each slot is
    still a set a person built: the item suits the spread, the moves suit the
    attacking stat, and none of it is the 37.7%-no-STAB noise a random team is.

    What recombining loses is the team's own idea -- a sand team's Tyranitar and
    its sweepers arrive separately -- so this is a distribution of good Pokemon
    rather than of good teams. That is the trade, and it is deliberate.
    """
    return tuple(pokemon
                 for party in ranker_parties(path)
                 for pokemon in party.team)


def ranker_team(
    dex: Dex,
    regulation: Regulation,
    cursor: RngCursor,
    battle_format: str = "singles",
    path: str | None = None,
) -> Team:
    """Six slots drawn from the ranker pool, legal by construction.

    Species Clause binds on the base species and Item Clause on the item, so
    both are held as we draw rather than checked afterwards -- rejection
    sampling over a pool this small would spend most of its time rejecting.
    """
    registered, _ = regulation.bring_select(battle_format)
    pool = list(ranker_slots(path))
    if len(pool) < registered:
        raise ValueError(f"only {len(pool)} ranker slots, need {registered}")

    team: list[PokemonSet] = []
    bases: set[str] = set()
    items: set[str] = set()
    for index in cursor.shuffled(list(range(len(pool)))):
        pokemon = pool[index]
        base = base_species_of(dex, pokemon.species)
        if base in bases or (pokemon.item and pokemon.item in items):
            continue
        team.append(pokemon)
        bases.add(base)
        if pokemon.item:
            items.add(pokemon.item)
        if len(team) == registered:
            return tuple(team)
    raise ValueError("the ranker pool cannot fill a legal team")


#: How a caller says which distribution it wants.
#:
#: ``"parties"`` also accepts a subset after a colon -- ``"parties:10,14,17,7"``
#: -- which is how the team curriculum is written. See ``party_team``.
TEAM_SOURCES = ("random", "ranker", "parties")


def make_team(
    dex: Dex,
    regulation: Regulation,
    cursor: RngCursor,
    battle_format: str = "singles",
    source: str = "random",
    options: RandomTeamOptions = RandomTeamOptions(),
) -> Team:
    """One team, from whichever distribution was asked for."""
    if source == "ranker":
        return ranker_team(dex, regulation, cursor, battle_format)
    if source == "parties" or source.startswith("parties:"):
        _, _, listed = source.partition(":")
        return party_team(dex, regulation, cursor, battle_format,
                          _party_indices(listed))
    if source != "random":
        raise ValueError(f"unknown team source {source!r}; expected {TEAM_SOURCES}")
    return random_team(dex, regulation, cursor, battle_format, options)


def parse_team_source(value: str) -> str:
    """Validate a ``--teams`` value now, rather than inside a worker later.

    ``argparse`` cannot express ``parties:10,14,17,7`` as a ``choices`` list, and
    a bad party index that only surfaces once ten spawned processes are three
    minutes into a run is a bad trade for the brevity.
    """
    if value in ("random", "ranker", "parties"):
        return value
    if value.startswith("parties:"):
        _party_indices(value.partition(":")[2])
        return value
    raise ValueError(f"unknown team source {value!r}; expected {TEAM_SOURCES} "
                     f"or parties:<comma separated indices>")


def _party_indices(listed: str) -> tuple[int, ...] | None:
    """``"10,14,17,7"`` to indices; empty means every imported party."""
    if not listed.strip():
        return None
    try:
        picked = tuple(int(one) for one in listed.split(",") if one.strip())
    except ValueError:
        raise ValueError(f"party subset {listed!r} is not a list of integers")
    if not picked:
        return None
    total = len(ranker_parties())
    for index in picked:
        if not 0 <= index < total:
            raise ValueError(f"party {index} is outside 0..{total - 1}")
    return picked


def party_team(
    dex: Dex,
    regulation: Regulation,
    cursor: RngCursor,
    battle_format: str = "singles",
    picked: tuple[int, ...] | None = None,
) -> Team:
    """One imported party, whole, as its author built it.

    The difference from ``ranker_team`` is the difference between AlphaZero's
    problem and this one. ``ranker_team`` draws six slots out of a hundred and
    twenty, so no two battles in a run ever share a matchup and the policy head
    is asked to generalise across roughly ten to the nineteen team pairings
    before it is asked to play well in any of them. Measured: held-out policy
    cross-entropy 0.315 above the target's own entropy with drawn teams and
    0.181 with a fixed four, and the sign of train-minus-validation flips with
    it -- drawn teams memorise, fixed teams generalise.

    Restricting ``picked`` narrows the curriculum further. Widening it back out
    is the experiment that says how much of the strength survives contact with
    a bigger team space.
    """
    parties = ranker_parties()
    choices = picked if picked is not None else tuple(range(len(parties)))
    return parties[choices[cursor.between(0, len(choices) - 1)]].team
