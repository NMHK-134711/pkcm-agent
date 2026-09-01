"""What one player knows, as opposed to what is true.

Principle (c) in docs/DESIGN.md, and the one the hybrid policy lives or dies on.
Pokemon is an imperfect-information game: the opponent's held items, its four
moves, its SP spread and which of its registered six it actually brought are all
hidden until something reveals them.

    BattleState                the truth, both sides' everything
    Observation.of(state, p)   only what ``p`` knows
    determinize(obs, ...)      one truth consistent with that observation

``Observation.of`` is a pure function of the state. That is the whole reason
``state.revealed`` exists: a wrapper that folded the event log would serve a
training loop perfectly well and be useless to a tree search, which arrives at a
node without having watched how it got there.

What is public, and why:

* **the registered six, both sides** -- team preview shows them before the
  battle starts.
* **which of them were brought** -- hidden. Learned one at a time, as each
  Pokemon is sent out.
* **HP and every other stat** -- our own exactly, because the game shows us our
  own Pokemon's numbers; the opponent's HP as a fraction, which is what a health
  bar shows, and their other stats not at all.
* **status, stat stages, most volatiles** -- public once applied; they are
  announced.
* **moves** -- our own fully; the opponent's only the ones we have watched it
  use.
* **item and ability** -- our own; the opponent's only once something announced
  them (Intimidate firing, a berry being eaten, Knock Off).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pkcm.engine.actions import Action
from pkcm.engine.state import BattleState, Phase, legal_actions

if TYPE_CHECKING:  # pragma: no cover
    from pkcm.engine.rng import RngCursor

#: Opponent HP is reported to this many steps, which is roughly what a health
#: bar resolves to. Reporting it exactly would hand over the SP spread.
HP_BAR_STEPS = 48


@dataclass(frozen=True, slots=True)
class KnownPokemon:
    """One Pokemon as the observer sees it.

    ``None`` means *unknown*, not *absent*. A field that is ``None`` for the
    opponent and populated for ourselves is hidden information, and every one
    of them is something ``determinize`` has to invent.
    """

    #: Index into the side's brought party.
    slot: int
    #: Where it is standing, or ``None`` if it is on the bench.
    position: int | None
    species_id: str | None
    hp_fraction: float
    #: Exact HP, ours only.
    hp: int | None
    max_hp: int | None
    status: str | None
    boosts: tuple[int, ...]
    volatiles: tuple[str, ...]
    #: Ours: all four. Theirs: only what we have watched it use.
    moves: tuple[str, ...]
    #: Ours only -- PP is not shown for the opponent.
    pp: tuple[int, ...] | None
    item: str | None
    item_known: bool
    ability: str | None
    ability_known: bool
    fainted: bool
    #: Our own six stats, exactly. ``None`` for the opponent, whose SP spread
    #: and nature are hidden -- ``pkcm.envs.analysis`` brackets those from the
    #: base stats instead, which is what a player does.
    stats: tuple[int, ...] | None = None
    #: Turns left on our own sleep or freeze. ``None`` for the opponent: the
    #: game shows that a Pokemon is asleep, never for how much longer.
    status_turns: int | None = None
    #: How many turns we have watched this status last. Public for both sides,
    #: and the only handle we have on how much longer *theirs* will go.
    status_elapsed: int | None = None
    #: Clean hits this Pokemon landed on *our* side: tuples of
    #: ``(move_id, damage, lethal, attacker_forme, defender_stats,
    #: defender_types)``. ``lethal`` flags a knockout, whose number is a floor
    #: rather than an exact roll; ``attacker_forme`` is the forme that swung,
    #: which for a stone-holder is not always the forme standing there now.
    #: Only filled for the opponent's Pokemon, and only with hits whose damage
    #: we read off our own HP bar -- an exact integer. The defender snapshot is
    #: ours to give: it is our own set. ``pkcm.envs.belief`` inverts the damage
    #: formula over these to eliminate candidate sets, which is what a person
    #: does the moment a hit lands harder than a bulky spread allows.
    hits_on_us: tuple = ()
    #: What it has already used up, when we watched that happen. ``item`` is
    #: what it holds *now*, which after a berry is nothing -- and collapsing the
    #: two throws away the most identifying thing about a set. Harvest and
    #: Recycle are why the difference is mechanical and not only informational.
    #:
    #: The two lines above and below met in a merge and belong together: a
    #: consumed Sitrus Berry is one more constraint the belief can narrow by,
    #: the same way a damage integer is.
    consumed_item: str | None = None

    @property
    def revealed(self) -> bool:
        """Has this one been seen at all? False for an unplayed bench slot."""
        return self.species_id is not None


@dataclass(frozen=True, slots=True)
class RegisteredSet:
    """One of *our* registered six, as we know it -- which is completely."""

    species_id: str
    moves: tuple[str, ...]
    item: str | None
    ability: str
    stats: tuple[int, ...]
    #: The build, not just its result. Needed because a Pokemon holding its own
    #: Mega Stone is scored as the forme it becomes, and the Mega's stats cannot
    #: be recovered from the base forme's -- see ``analysis.fought_as``.
    sp: tuple[int, ...] = ()
    nature: str = ""


@dataclass(frozen=True, slots=True)
class Observation:
    """Everything ``player`` may condition on, and nothing else."""

    player: int
    phase: Phase
    turn: int
    #: The six each side registered. Public from team preview, both sides.
    registered: tuple[tuple[str, ...], tuple[str, ...]]
    own: tuple[KnownPokemon, ...]
    foe: tuple[KnownPokemon, ...]
    own_conditions: tuple[tuple[str, int], ...]
    foe_conditions: tuple[tuple[str, int], ...]
    weather: str | None
    weather_turns: int
    terrain: str | None
    terrain_turns: int
    rooms: tuple[tuple[str, int], ...]
    #: One tuple of legal actions per field position.
    legal: tuple[tuple[Action, ...], ...]
    #: Mega Evolution already spent, per side. Public -- it is announced.
    mega_used: tuple[bool, bool]
    winner: int | None
    #: **Our own** registered six, in full: moves, item, ability and stats.
    #:
    #: Ours to read at any time, and at team preview it is the only thing that
    #: separates two teams of the same six species. Without it, a physical
    #: Garchomp carrying Earthquake and a special one carrying Water Gun encode
    #: to byte-identical observations while the pick they justify is different
    #: -- so a policy asked to pick could not do better than chance, and
    #: measured, it did not: 5% top-1 agreement on a 24-way choice.
    #:
    #: The opponent's six stay species-only above. That is what preview shows.
    own_sets: tuple["RegisteredSet", ...] = ()

    @staticmethod
    def of(state: BattleState, player: int) -> "Observation":
        foe = 1 - player
        positions = state.config.active_count
        return Observation(
            player=player,
            phase=state.phase,
            turn=state.turn,
            registered=(
                tuple(p.species.id for p in state.parties[player]),
                tuple(p.species.id for p in state.parties[foe]),
            ),
            own_sets=tuple(
                RegisteredSet(
                    species_id=pokemon.species.id,
                    moves=tuple(move.id for move in pokemon.moves),
                    item=pokemon.item,
                    ability=pokemon.ability,
                    stats=tuple(pokemon.stats),
                    sp=tuple(pokemon.set.sp),
                    nature=pokemon.set.nature,
                )
                for pokemon in state.parties[player]
            ),
            own=tuple(_own_view(state, player, slot)
                      for slot in range(len(state.sides[player].hp))),
            foe=tuple(_foe_view(state, foe, slot)
                      for slot in range(len(state.sides[foe].hp))),
            own_conditions=tuple(sorted(state.sides[player].conditions.items())),
            foe_conditions=tuple(sorted(state.sides[foe].conditions.items())),
            weather=state.field.weather,
            weather_turns=state.field.weather_turns,
            terrain=state.field.terrain,
            terrain_turns=state.field.terrain_turns,
            rooms=tuple(sorted(state.field.rooms.items())),
            legal=tuple(legal_actions(state, player, position)
                        for position in range(positions)),
            mega_used=(state.mega_used[player], state.mega_used[foe]),
            winner=state.winner,
        )

    @property
    def finished(self) -> bool:
        return self.phase is Phase.FINISHED

    def action_mask(self, position: int = 0) -> tuple[Action, ...]:
        return self.legal[position] if position < len(self.legal) else ()


# --------------------------------------------------------------------------- #
# Building the two views
# --------------------------------------------------------------------------- #


def _shared(state: BattleState, side: int, slot: int) -> dict:
    """The half that is public either way: where it stands, how hurt it is."""
    side_state = state.sides[side]
    return {
        "slot": slot,
        "position": side_state.position_of(slot),
        "status": side_state.status[slot],
        "boosts": tuple(side_state.boosts[slot]),
        "volatiles": tuple(sorted(side_state.volatiles[slot])),
        "fainted": side_state.is_fainted(slot),
    }


def _own_view(state: BattleState, side: int, slot: int) -> KnownPokemon:
    side_state = state.sides[side]
    maximum = state.pokemon(side, slot).max_hp
    return KnownPokemon(
        **_shared(state, side, slot),
        species_id=state.species_id(side, slot),
        hp=side_state.hp[slot],
        max_hp=maximum,
        hp_fraction=side_state.hp[slot] / maximum if maximum else 0.0,
        moves=tuple(move.id for move in state.moves(side, slot)),
        pp=tuple(side_state.pp[slot]),
        item=state.item_id(side, slot),
        item_known=True,
        consumed_item=_consumed(state, side, slot),
        ability=state.ability_id(side, slot),
        ability_known=True,
        stats=tuple(state.stats(side, slot)),
        status_turns=side_state.status_data[slot].get("turns"),
        status_elapsed=_elapsed(state, side, slot),
    )


def _hits_by(state: BattleState, attacker_side: int, attacker_slot: int,
             observer: int) -> tuple:
    """The clean hits one foe landed on the observer, priced-ready.

    The damage integers are exact because the HP that moved was the
    observer's own. The defender snapshot -- species, exact stats, live types
    -- is the observer's own set, theirs to hand to the pricer. The forme in
    the ledger entry is folded away here: the belief re-derives each
    candidate's forme from its item, and the KnownPokemon's ``species_id``
    already names what was seen on the field.
    """
    found = []
    for entry in state.observed_hits:
        (d_side, _d_slot, a_side, a_slot, forme, move_id, dealt, lethal,
         _d_species, d_stats, d_types) = entry
        if a_side != attacker_side or a_slot != attacker_slot:
            continue
        if d_side != observer:
            continue
        # Both snapshots were taken by the engine at the moment the hit landed
        # -- reading the state now would lie about anyone who changed forme
        # since. The attacker's forme rides along per hit for the same reason:
        # a number thrown before Mega Evolving answers to base stats however
        # the attacker looks now. ``lethal`` marks a truncated number: a
        # knockout shows what the HP bar absorbed, not what was rolled.
        found.append((move_id, dealt, lethal, forme, d_stats, d_types))
    return tuple(found)


def _foe_view(state: BattleState, side: int, slot: int) -> KnownPokemon:
    """The opponent, with everything unannounced left as ``None``.

    A Pokemon that has never been sent out has no species here even though the
    state knows perfectly well what it is -- which is the point.
    """
    side_state = state.sides[side]
    seen = state.revealed[side]
    on_field = slot in seen.species
    maximum = state.pokemon(side, slot).max_hp

    fraction = side_state.hp[slot] / maximum if maximum else 0.0
    return KnownPokemon(
        **_shared(state, side, slot),
        species_id=state.species_id(side, slot) if on_field else None,
        hp=None,
        max_hp=None,
        hp_fraction=round(fraction * HP_BAR_STEPS) / HP_BAR_STEPS if on_field else 1.0,
        moves=tuple(sorted(seen.moves_of(slot))),
        pp=None,
        item=state.item_id(side, slot) if slot in seen.items else None,
        item_known=slot in seen.items,
        consumed_item=_consumed(state, side, slot) if slot in seen.items else None,
        ability=state.ability_id(side, slot) if slot in seen.abilities else None,
        ability_known=slot in seen.abilities,
        status_elapsed=_elapsed(state, side, slot),
        hits_on_us=_hits_by(state, side, slot, observer=1 - side),
    )


def _consumed(state: BattleState, side: int, slot: int) -> str | None:
    """The item this one used up, if it has used one.

    The engine marks a spent item by overriding it to ``None`` while the set
    keeps the original -- which is exactly what Recycle reads to give it back.
    Reading it the same way here means an opponent that ate its Sitrus Berry
    stops looking, to the belief, like an opponent known to hold nothing.
    """
    override = state.override(side, slot)
    if "item" in override and override["item"] is None:
        return state.pokemon(side, slot).item
    return None


def _elapsed(state: BattleState, side: int, slot: int) -> int | None:
    """How many turns this status has been up. ``None`` if it has none."""
    since = state.revealed[side].status_since.get(slot)
    return None if since is None else max(0, state.turn - since)


# --------------------------------------------------------------------------- #
# Determinizing
# --------------------------------------------------------------------------- #


def determinize(observation: Observation, truth: BattleState,
                cursor: "RngCursor", belief: bool = False) -> BattleState:
    """One full state consistent with what the observer knows.

    Search needs a concrete state to roll out from, and an observation is not
    one -- it has holes where the opponent's secrets are. This fills them with a
    sample that contradicts nothing observed.

    ``truth`` supplies the *shape*: the config, our own side, the turn, the
    field. Every hidden field of the opponent's is resampled rather than copied,
    and that distinction is the whole value of the function. A determinization
    that quietly kept the real species would produce a search that plays
    perfectly against the team actually in front of it and has learned nothing
    transferable -- and it would look like a very strong search.

    What gets resampled, and from what:

    * **species of an unrevealed Pokemon** -- from their registered six, minus
      the ones already seen. Team preview is why that pool is known at all.
    * **moves** -- the ones we have watched, topped up from what the species can
      learn.
    * **ability** -- from the ones that species may have.
    * **item** -- from what the format allows, honouring the Item Clause against
      the items we have already seen them use.
    * **SP spread and nature** -- a legal random spread. The bracket in
      ``pkcm.envs.analysis`` is the honest summary of this uncertainty; here it
      has to be resolved into one concrete answer.

    With ``belief``, a whole set is drawn from the ranker pool first and the
    per-field sampling below is only the fallback -- see ``pkcm.envs.belief``.
    Every field above stays *consistent* under either path; what changes is
    whether it is also *plausible*, and whether watching a move go off narrows
    anything.
    """
    from pkcm.engine.legality import (
        champions_items,
        learnable_moves,
        random_sp,
        registrable_abilities,
    )
    from pkcm.engine.pokemon import MAX_MOVES, PokemonSet, compile_team
    from pkcm.engine.stats import NATURES

    dex = truth.config.dex
    foe = 1 - observation.player
    sampled = list(truth.parties[foe])
    side = truth.sides[foe]

    # The pool an unrevealed Pokemon is drawn from: their registered six, less
    # whatever we have already watched come out.
    seen_species = {known.species_id for known in observation.foe if known.species_id}
    unseen = [species for species in observation.registered[1]
              if species not in seen_species]
    taken_items = {known.item for known in observation.foe
                   if known.item_known and known.item}

    for known in observation.foe:
        if known.slot >= len(side.selection):
            continue
        party_index = side.selection[known.slot]

        if known.species_id is not None:
            species_id = known.species_id
        elif unseen:
            species_id = unseen.pop(cursor.between(0, len(unseen) - 1))
        else:
            continue  # nothing left to draw from; leave the slot as it was

        if belief:
            from pkcm.envs.belief import sample as sample_set

            drawn = sample_set(species_id, known, cursor)
            if drawn is not None and (drawn.item is None
                                      or drawn.item not in taken_items
                                      or known.item_known):
                sampled[party_index] = compile_team(dex, (drawn,))[0]
                if drawn.item:
                    taken_items.add(drawn.item)
                continue

        moves = list(known.moves)
        for candidate in _shuffled(sorted(learnable_moves(dex, species_id)), cursor):
            if len(moves) >= MAX_MOVES:
                break
            if candidate not in moves:
                moves.append(candidate)

        ability = known.ability
        if ability is None:
            options = registrable_abilities(dex.species[species_id])
            ability = options[cursor.between(0, len(options) - 1)]

        if known.item_known:
            # The set holds what it started with; the cloned overrides below
            # still say it has been spent, so the determinization ends up in
            # exactly the state the real battle is in.
            item = known.consumed_item or known.item
        else:
            item = _sample_item(dex, cursor, taken_items)
            if item is not None:
                taken_items.add(item)

        sampled[party_index] = compile_team(dex, (PokemonSet(
            species=species_id,
            ability=ability,
            moves=tuple(moves[:MAX_MOVES]) or (moves + ["struggle"])[:1],
            item=item,
            nature=_sample_nature(cursor),
            sp=random_sp(cursor),
        ),))[0]

    guess = truth.clone()
    parties = list(guess.parties)
    parties[foe] = tuple(sampled)
    guess.parties = tuple(parties)
    _rescale_hp(guess, foe, observation)
    return guess


def _sample_nature(cursor: "RngCursor") -> str:
    from pkcm.engine.stats import NATURES

    names = sorted(NATURES)
    return names[cursor.between(0, len(names) - 1)]


def _sample_item(dex, cursor: "RngCursor", taken: set[str]) -> str | None:
    """A held item the Item Clause still allows them.

    Mega stones are excluded: holding one is visible the moment it is used, and
    an unrevealed Pokemon carrying one that never fires would make the search
    plan around a Mega Evolution that cannot happen.
    """
    from pkcm.engine.legality import champions_items

    pool = [item for item in sorted(champions_items())
            if item not in taken and not dex.items[item].mega_stone]
    if not pool:
        return None
    return pool[cursor.between(0, len(pool) - 1)]


def _rescale_hp(state: BattleState, side: int, observation: Observation) -> None:
    """Keep the HP *fraction* we observed after swapping the species underneath.

    HP is stored as an absolute number and maximums differ by species, so a
    resampled Pokemon would otherwise arrive at the wrong health -- sometimes
    above its own maximum.
    """
    for known in observation.foe:
        if known.slot >= len(state.sides[side].hp):
            continue
        maximum = state.pokemon(side, known.slot).max_hp
        state.sides[side].hp[known.slot] = (
            0 if known.fainted else max(1, round(known.hp_fraction * maximum))
        )


def _shuffled(items: list[str], cursor: "RngCursor") -> list[str]:
    picked = list(items)
    for index in range(len(picked) - 1, 0, -1):
        swap = cursor.between(0, index)
        picked[index], picked[swap] = picked[swap], picked[index]
    return picked
