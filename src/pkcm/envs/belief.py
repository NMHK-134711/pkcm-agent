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


def use_pool(source: str) -> None:
    """Switch the pool. ``sets_by_species`` is cached, so this clears it."""
    global _SOURCE
    if source not in ("ranker", "invented"):
        raise ValueError(f"unknown belief pool {source!r}")
    _SOURCE = source
    sets_by_species.cache_clear()


@lru_cache(maxsize=1)
def sets_by_species() -> dict[str, tuple[PokemonSet, ...]]:
    """The sets the belief draws from, grouped by species."""
    from pkcm.engine.legality import ranker_slots

    grouped: dict[str, list[PokemonSet]] = defaultdict(list)
    for pokemon in ranker_slots():
        grouped[pokemon.species].append(pokemon)
    real = {species: tuple(found) for species, found in grouped.items()}
    return real if _SOURCE == "ranker" else _invented_like(real)


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


def consistent(candidate: PokemonSet, known) -> bool:
    """Could this set be the thing we have been watching?

    Only what the observation actually establishes is checked. ``item_known``
    and ``ability_known`` matter because a ``None`` there means *unseen*, not
    *absent* -- treating an unrevealed item as "no item" would throw away every
    candidate that holds one.
    """
    if known.species_id is not None and candidate.species != known.species_id:
        return False
    if not set(known.moves).issubset(candidate.moves):
        return False
    if known.item_known and known.item != candidate.item:
        return False
    if known.ability_known and known.ability != candidate.ability:
        return False
    return True


def candidates(species_id: str, known) -> tuple[PokemonSet, ...]:
    """The ranker sets for this species that fit what we have seen."""
    pool = sets_by_species().get(species_id)
    if not pool:
        return ()
    return tuple(one for one in pool if consistent(one, known))


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
