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


@lru_cache(maxsize=1)
def sets_by_species() -> dict[str, tuple[PokemonSet, ...]]:
    """Every ranker set this project has imported, grouped by species."""
    from pkcm.engine.legality import ranker_slots

    grouped: dict[str, list[PokemonSet]] = defaultdict(list)
    for pokemon in ranker_slots():
        grouped[pokemon.species].append(pokemon)
    return {species: tuple(found) for species, found in grouped.items()}


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
