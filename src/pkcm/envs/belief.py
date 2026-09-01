"""What the opponent is probably running, given what they have shown.

``determinize`` has to invent the opponent's hidden fields, and it used to
invent each one independently and uniformly: a move drawn from the sixty that
species can learn, an item from all 147 the format allows, a nature from
twenty-one. Nothing about that is *inconsistent* with what we have seen -- it is
simply nothing like a real set. The search then spends its budget planning
against opponents nobody would bring.

The ranker pool says what people actually bring. Over its 120 slots, **rankers
use 29 of the 147 legal items**, and four of those account for half of them.
Moves are concentrated the same way.

So this samples a **whole set** rather than field by field. Two things fall out
of that which per-field sampling cannot give:

* **The set is coherent.** A Choice Scarf comes with the spread that wants one.
  Sampling an item and a nature independently produces neither.
* **It conditions on the battle so far.** A set is only a candidate if every
  move we have watched them use is on it, and if the item and ability we have
  identified match. Watching Earthquake go off narrows the belief; under
  per-field sampling it narrowed nothing, because the other three moves were
  redrawn from sixty regardless.

Where the pool has nothing -- 42 species of the roster's 235 are in it --
``determinize`` keeps its old behaviour. That is a floor, not a fallback to be
embarrassed about: a species nobody brought is one the search will rarely meet.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

from pkcm.engine.pokemon import PokemonSet


#: Which pool the belief samples from. ``"ranker"`` is what people bring;
#: ``"invented"`` is the same shape filled with legal sets nobody plays.
#:
#: The second is an ablation and exists to answer one question: how much of
#: belief's gain is *knowing what people run*, and how much is the machinery
#: around it -- sampling a whole coherent set rather than field by field, and
#: narrowing it by every move we have watched.
#:
#: That distinction decides how much a patch costs, and it has been measured.
#: Four hundred games each, against sampling the hidden fields uniformly:
#:
#:     ranker pool     68.6% [63.9, 73.0]   separably stronger
#:     invented pool   51.3% [46.4, 56.1]   not separable
#:
#: Same 42 species, same 120 sets, same count each -- only the contents differ.
#: **The whole gain is knowing what people actually run.** Sampling coherent
#: sets and narrowing them by what we have watched, on a pool of plausible but
#: unplayed sets, is worth nothing measurable over drawing each field at
#: random. Narrowing perfectly inside a wrong candidate set does not help.
#:
#: So a patch is a cliff, not a slope. The docstring above says a species the
#: pool has never seen "is one the search will rarely meet", and that is true;
#: what is not true is the reassurance it was offering. A species the pool
#: knows *wrongly* is as bad as one it does not know at all, and after a patch
#: that describes every species whose sets moved.
_SOURCE = "ranker"


#: The pools this can draw from.
#:
#: ``"imported"`` is the twenty parties this project started with, and it is
#: here to price the transcribed ones. Belief is worth eighteen points and the
#: shuffle test says all of it comes from the snapshot being *real*, so a
#: snapshot that grew from 42 species to 54 should be worth something -- but
#: more sets per species also spreads the candidates, and narrowing by watched
#: moves gets harder as the pool widens. That is a measurement, not a deduction.
POOLS = ("ranker", "imported", "invented")


def use_pool(source: str) -> None:
    """Switch the pool. ``sets_by_species`` is cached, so this clears it."""
    global _SOURCE
    if source not in POOLS:
        raise ValueError(f"unknown belief pool {source!r}; expected {POOLS}")
    _SOURCE = source
    sets_by_species.cache_clear()


@lru_cache(maxsize=1)
def sets_by_species() -> dict[str, tuple[PokemonSet, ...]]:
    """The sets the belief draws from, grouped by species."""
    from pkcm.engine.legality import PARTIES_PATH, ranker_slots

    slots = (ranker_slots() if _SOURCE != "imported"
             else ranker_slots(str(PARTIES_PATH)))
    grouped: dict[str, list[PokemonSet]] = defaultdict(list)
    for pokemon in slots:
        grouped[pokemon.species].append(pokemon)
    real = {species: tuple(found) for species, found in grouped.items()}
    return real if _SOURCE != "invented" else _invented_like(real)


def _invented_like(real: dict[str, tuple[PokemonSet, ...]]
                   ) -> dict[str, tuple[PokemonSet, ...]]:
    """The same pool shape, filled with legal sets nobody brings.

    Same species, same number of candidates each, so the narrowing has the same
    resolution to work with and only the *contents* differ. Anything the
    comparison finds is then about knowing what people run, not about having
    more or fewer candidates to choose between.
    """
    from pkcm.data.dex import load_dex
    from pkcm.engine.legality import random_set
    from pkcm.engine.rng import Rng

    dex = load_dex()
    cursor = Rng.from_seed(0xBE11EF).cursor()
    invented: dict[str, tuple[PokemonSet, ...]] = {}
    for species, found in real.items():
        made = []
        for _ in found:
            try:
                made.append(random_set(dex, species, cursor))
            except ValueError:
                pass
        # A species the random builder cannot serve keeps its real sets rather
        # than dropping out of the pool, which would change the shape.
        invented[species] = tuple(made) if made else found
    return invented


#: Flat item multipliers the pricer understands -- and it understands them by
#: calling the engine's own ``chain_modify`` with the engine's own constants,
#: because "1.3" is not what Life Orb does. Life Orb is 5325/4096ths rounded
#: half up, and pricing it as a float floor put the number 99 outside a roll
#: table whose engine really rolled 99. Anything not here prices as 1.0, and
#: the fallback in ``candidates`` keeps an unmodeled modifier from ever
#: emptying the pool. Choice Band and Specs do not exist in Champions.
#: Lowercase on purpose: the dex speaks lowercase types, and the first draft
#: of this table said "Ghost" -- so Spell Tag priced as nothing and the true
#: Aegislash was struck off by a hit it really threw. The invariant test
#: (tests/test_damage_inference.py) is what caught it.
_POWER_ITEMS = {
    "blackglasses": "dark", "spelltag": "ghost", "mysticwater": "water",
    "fairyfeather": "fairy", "charcoal": "fire", "miracleseed": "grass",
    "magnet": "electric", "sharpbeak": "flying", "softsand": "ground",
    "silkscarf": "normal", "hardstone": "rock", "silverpowder": "bug",
    "dragonfang": "dragon", "metalcoat": "steel", "twistedspoon": "psychic",
    "nevermeltice": "ice", "poisonbarb": "poison", "blackbelt": "fighting",
}
_ROLL_LOW, _ROLL_HIGH = 85, 100

#: The skin family, exactly as the engine's ``_ate`` handlers write it: a
#: Normal-type damaging move takes the ability's type and 4915/4096ths of its
#: power. Liquid Voice retypes sound moves to Water and boosts nothing.
_ATE = {"pixilate": "fairy", "refrigerate": "ice", "aerilate": "flying",
        "galvanize": "electric", "dragonize": "dragon"}


def _hit_rolls(candidate: PokemonSet, forme_seen: str, move_id: str,
               defender_stats, defender_types) -> tuple[int, ...] | None:
    """The sixteen damages this candidate could have rolled, or ``None``.

    ``None`` means "cannot price", never "impossible" -- an unpriceable hit
    must not eliminate anyone. Impossible is an empty membership test by the
    caller. The forme check is real information though: a candidate whose item
    is not the stone the seen Mega requires could not have been the attacker.
    """
    from pkcm.data.dex import load_dex
    from pkcm.engine.moves import (X1_2, X1_3, chain_modify, damage_base,
                                   damage_from_base)
    from pkcm.envs.analysis import fought_as

    dex = load_dex()
    move = dex.moves.get(move_id)
    if move is None or not move.base_power or move.is_status:
        return None
    from pkcm.engine.moves import VARIABLE_POWER, _touches_damage

    if move.raw.get("basePowerCallback") or move.raw.get("multihit") \
            or move.raw.get("damage") or move.id in VARIABLE_POWER:
        return None
    # Body Press and friends read a stat this formula does not. The recorder
    # keeps their hits out of the ledger; this guard keeps the pricer honest
    # should one arrive anyway.
    if move.raw.get("overrideOffensiveStat") \
            or move.raw.get("overrideOffensivePokemon") \
            or move.raw.get("overrideDefensiveStat"):
        return None

    # Price the forme that was actually on the field when the hit landed --
    # per hit, not per Pokemon. A stone-holder that had not Mega Evolved yet
    # swings with its base stats, and pricing it as the Mega it will become
    # eliminated Floette-Eternal for a Draining Kiss it genuinely threw.
    seen = dex.species.get(forme_seen)
    if seen is None:
        return None
    if seen.is_mega:
        # A Mega is only reachable through its stone, so the forme doubles as
        # an item test: a candidate holding anything else is struck off.
        forme, stats, types = fought_as(dex, candidate.species, candidate.item,
                                        candidate.ability, candidate.sp,
                                        candidate.nature)
        if forme != forme_seen:
            return ()
        fighting_ability = dex.species[forme].abilities[0]
    elif seen.base_species == dex.species[candidate.species].base_species:
        # The forme that swung, whatever moved it there -- Aegislash attacks
        # from Blade, a stone-holder before Mega Evolving attacks from base --
        # priced with that forme's base stats under the candidate's spread.
        from pkcm.engine.stats import compute_stats, get_nature

        stats = tuple(compute_stats(seen.base_stats, candidate.sp,
                                    get_nature(candidate.nature)))
        types = seen.types
        fighting_ability = candidate.ability
    else:
        return ()          # a forme this species line does not contain

    # A candidate whose ability rewrites damage prices differently than this
    # formula says. The doubled-Attack pair and the skin family are modeled
    # exactly below; any other damage-touching ability makes the hit
    # unreadable *for this candidate* -- unreadable, not impossible, so the
    # candidate survives rather than being wrongly struck off.
    from pkcm.engine.moves import PRICED_ATTACKER_ABILITIES

    if fighting_ability not in PRICED_ATTACKER_ABILITIES \
            and _touches_damage("ability", fighting_ability):
        return None
    if fighting_ability in ("hugepower", "purepower") \
            and forme_seen == candidate.species:
        stats = tuple(value * 2 if index == 1 else value
                      for index, value in enumerate(stats))

    # The skin family rewrites the move before anything else looks at it, the
    # way the engine's modify_move hook does -- so STAB, effectiveness and
    # the type-boost items below all see the rewritten move. This is also
    # where damage inference starts identifying *abilities*: a Liquid Voice
    # Primarina and a Torrent one throw visibly different Hyper Voices.
    move_type = move.type
    power = move.base_power
    if fighting_ability in _ATE:
        if move_type == "normal" and move.category != "Status":
            move_type = _ATE[fighting_ability]
            power = chain_modify(power, X1_2)
    elif fighting_ability == "normalize":
        if move_type != "normal" and move.category != "Status":
            move_type = "normal"
            power = chain_modify(power, X1_2)
    elif fighting_ability == "liquidvoice" and "sound" in move.flags:
        move_type = "water"

    effectiveness = dex.type_chart.multiplier(move_type, tuple(defender_types))
    if effectiveness == 0:
        return None        # the hit landed, so this pricing is wrong somewhere

    physical = move.category == "Physical"
    attack = stats[1] if physical else stats[3]
    defense = defender_stats[2] if physical else defender_stats[4]

    item = candidate.item
    if _POWER_ITEMS.get(item) == move_type:
        power = chain_modify(power, X1_2)

    stab = move_type in types
    base = damage_base(power=power, attack=attack, defense=defense,
                       crit=False, spread=False)
    rolls = []
    for roll in range(_ROLL_LOW, _ROLL_HIGH + 1):
        damage = damage_from_base(base, roll, stab=stab,
                                  effectiveness=effectiveness)
        if item == "lifeorb":
            damage = chain_modify(damage, X1_3)
        elif item == "expertbelt" and effectiveness > 1:
            damage = chain_modify(damage, X1_2)
        rolls.append(max(1, damage))
    return tuple(rolls)


def survives_hits(candidate: PokemonSet, known) -> bool:
    """Could this set have thrown every number we watched land?

    A lethal hit is a floor, not an exact roll -- the knockout truncated it --
    so it asks only that some roll reaches the number. Everything else must
    be rolled exactly.
    """
    for move_id, damage, lethal, attacker_forme, defender_stats, \
            defender_types in getattr(known, "hits_on_us", ()):
        # The ledger's forme, not the current one: a hit thrown before Mega
        # Evolving answers to base stats however the attacker looks now.
        rolls = _hit_rolls(candidate, attacker_forme, move_id,
                           defender_stats, defender_types)
        if rolls is None:
            continue
        if lethal:
            if max(rolls) < damage:
                return False
        elif damage not in rolls:
            return False
    return True


def consistent(candidate: PokemonSet, known) -> bool:
    """Could this set be the thing we have been watching?

    Only what the observation actually establishes is checked. ``item_known``
    and ``ability_known`` matter because a ``None`` there means *unseen*, not
    *absent* -- treating an unrevealed item as "no item" would throw away every
    candidate that holds one.
    """
    if known.species_id is not None and candidate.species != known.species_id:
        # A Mega on the field is still the base set on the roster. Requiring
        # the ids to match verbatim emptied the pool the moment an opponent
        # Mega Evolved -- and from there to the end of the battle the
        # determinizer fell back to uniform random, for the species the whole
        # team was built around. Seeing the Mega also *reveals* the item: only
        # one stone gets there, so a candidate holding anything else is out.
        from pkcm.data.dex import load_dex

        dex = load_dex()
        seen = dex.species.get(known.species_id)
        if seen is None or not seen.is_mega \
                or seen.base_species != candidate.species:
            return False
        if seen.required_item and candidate.item != seen.required_item:
            return False
    if not set(known.moves).issubset(candidate.moves):
        return False
    # What it was, not what is left of it. A Pokemon that has eaten its Sitrus
    # Berry holds nothing, and matching on that would throw away every set built
    # around the berry -- which is to say the set we just identified.
    if known.item_known and (known.consumed_item or known.item) != candidate.item:
        return False
    if known.ability_known and known.ability != candidate.ability:
        return False
    return True


def candidates(species_id: str, known) -> tuple[PokemonSet, ...]:
    """The ranker sets for this species that fit what we have seen.

    Two passes. Moves, item and ability first -- those are hard facts. Then
    the damage numbers, **with a fallback**: if every remaining candidate
    fails the numbers, the numbers are ignored for this draw. That is the
    guard against the pricer's blind spots -- an ability multiplier it does
    not model would otherwise empty the pool and take the true set with it.
    Measured offline, one clean number eliminates 35.6% of move-compatible
    rivals and three leave 2.4 standing, so the filter earns its place; the
    fallback just keeps it honest about what it cannot price.
    """
    pool = sets_by_species().get(species_id)
    if not pool:
        # A Mega Evolved opponent asks under its Mega id; the pool files sets
        # under the base species it registered as.
        from pkcm.data.dex import load_dex

        seen = load_dex().species.get(species_id)
        if seen is not None and seen.is_mega:
            pool = sets_by_species().get(seen.base_species)
    if not pool:
        return ()
    watched = tuple(one for one in pool if consistent(one, known))
    if not watched or not getattr(known, "hits_on_us", ()):
        return watched
    priced = tuple(one for one in watched if survives_hits(one, known))
    return priced or watched


def sample(species_id: str, known, cursor) -> PokemonSet | None:
    """One of them, or ``None`` when the pool cannot answer.

    ``None`` is the honest reply in two cases and they are worth telling apart
    only in a report: nobody in the pool plays this species, or everyone who
    does contradicts what we have watched. Either way the caller falls back.
    """
    found = candidates(species_id, known)
    if not found:
        return None
    return found[cursor.between(0, len(found) - 1)]
