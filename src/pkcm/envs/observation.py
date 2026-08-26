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

    @property
    def revealed(self) -> bool:
        """Has this one been seen at all? False for an unplayed bench slot."""
        return self.species_id is not None


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
        ability=state.ability_id(side, slot),
        ability_known=True,
        stats=tuple(state.stats(side, slot)),
    )


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
        ability=state.ability_id(side, slot) if slot in seen.abilities else None,
        ability_known=slot in seen.abilities,
    )


# --------------------------------------------------------------------------- #
# Determinizing
# --------------------------------------------------------------------------- #


def determinize(observation: Observation, truth: BattleState,
                cursor: "RngCursor") -> BattleState:
    """One full state consistent with what the observer knows.

    Search needs a concrete state to roll out from, and the observation is not
    one -- it has holes where the opponent's secrets are. This fills them in
    with a sample that contradicts nothing observed.

    ``truth`` supplies the *shape*: the config, the parties, the turn. Its
    hidden fields are the ones being resampled, so passing the real state is
    not cheating as long as only the observable parts survive -- which is what
    the assertions here are for.
    """
    from pkcm.engine.legality import learnable_moves, registrable_abilities
    from pkcm.engine.pokemon import MAX_MOVES, PokemonSet, compile_team

    dex = truth.config.dex
    foe = 1 - observation.player
    sampled = list(truth.parties[foe])

    for known in observation.foe:
        if known.slot >= len(truth.sides[foe].selection):
            continue
        party_index = truth.sides[foe].selection[known.slot]
        actual = truth.parties[foe][party_index]

        # Species is public once seen; before that the registered six bound it,
        # and we keep the real one rather than reshuffling the whole selection.
        species_id = known.species_id or actual.species.id
        pool = sorted(learnable_moves(dex, species_id))

        moves = list(known.moves)
        for candidate in _shuffled(pool, cursor):
            if len(moves) >= MAX_MOVES:
                break
            if candidate not in moves:
                moves.append(candidate)

        ability = known.ability
        if ability is None:
            options = registrable_abilities(dex.species[species_id])
            ability = options[cursor.between(0, len(options) - 1)]

        sampled[party_index] = compile_team(dex, (PokemonSet(
            species=species_id,
            ability=ability,
            moves=tuple(moves[:MAX_MOVES]),
            item=known.item if known.item_known else actual.item,
            nature=actual.set.nature,
            sp=actual.set.sp,
        ),))[0]

    guess = truth.clone()
    parties = list(guess.parties)
    parties[foe] = tuple(sampled)
    guess.parties = tuple(parties)
    return guess


def _shuffled(items: list[str], cursor: "RngCursor") -> list[str]:
    picked = list(items)
    for index in range(len(picked) - 1, 0, -1):
        swap = cursor.between(0, index)
        picked[index], picked[swap] = picked[swap], picked[index]
    return picked
