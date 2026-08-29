"""The real game, mirrored closely enough for the search to run on it.

The search needs a ``BattleState``. A real game gives us an opponent whose six
we can read off team preview and nothing else -- no spreads, no items, no moves
until they are used, and not even which three they brought until they walk out.

The trick is that **the search never reads this state's opponent.** ``MCTS``
takes ``Observation.of(state, us)`` and then ``determinize``s it, and
determinize resamples every hidden field of theirs from ``state.revealed``. So
the opponent's half of this state is a *placeholder*: it exists to give the
engine something to step, and its secrets are thrown away before any thinking
happens. What matters is that the placeholder is consistent with what has
actually been seen, and that is what this class maintains.

Three things arrive from outside, because only the person watching the screen
knows them:

* **What the opponent did.** ``report`` takes a move or a switch and rewrites
  the placeholder so the engine can step it -- teaching it a move it did not
  have, or swapping an unrevealed slot for the Pokemon that actually appeared.
* **How much HP is left.** Our own damage rolls are not theirs, so after every
  step the numbers are wrong by a little and sometimes by a lot. ``observe``
  overwrites them with what the bars say.
* **Status.** Same reason, and it is one word.

Drift is not an error case here, it is the normal case. The design goal is that
correcting it costs a couple of keystrokes rather than a re-entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pkcm.data.dex import Dex
from pkcm.engine.actions import Action
from pkcm.engine.battle import step
from pkcm.engine.pokemon import (
    MAX_MOVES,
    PokemonSet,
    Team,
    compile_team,
    max_pp,
)
from pkcm.engine.rng import Rng, RngCursor
from pkcm.engine.state import (
    BattleConfig,
    BattleState,
    Phase,
    legal_actions,
    new_battle,
)


US, THEM = 0, 1


class MirrorError(Exception):
    """The report cannot be reconciled with the game we think we are watching."""


@dataclass
class Mirror:
    """One live game, ours to advise on and theirs to tell us about."""

    config: BattleConfig
    state: BattleState
    cursor: RngCursor
    #: Their registered six, in the order they were entered. Indexes
    #: ``state.parties[THEM]``.
    their_six: tuple[str, ...]
    log: list[str] = field(default_factory=list)

    # -- starting ----------------------------------------------------------- #

    @staticmethod
    def begin(dex: Dex, regulation, our_team: Team, their_six,
              battle_format: str = "singles", seed: int = 0) -> "Mirror":
        """A game at team preview, with their six as read off the screen.

        Their sets are placeholders drawn from the ranker pool where it knows
        the species, because a plausible placeholder makes the mirrored damage
        roughly right and every point it is right by is a correction the person
        at the keyboard does not have to type. It is never used as knowledge --
        see the module docstring.
        """
        config = BattleConfig(dex=dex, regulation=regulation,
                              battle_format=battle_format)
        cursor = Rng.from_seed(seed).cursor()
        their_six = tuple(their_six)
        registered = config.registered
        if len(their_six) != registered:
            raise MirrorError(
                f"team preview shows {registered} Pokemon, got {len(their_six)}")
        for species in their_six:
            if species not in dex.species:
                raise MirrorError(f"no such species: {species!r}")

        their_team = tuple(_placeholder(dex, species, cursor)
                           for species in their_six)
        state = new_battle(config, (tuple(our_team), their_team), seed=seed)
        return Mirror(config=config, state=state, cursor=cursor,
                      their_six=their_six)

    # -- reading ------------------------------------------------------------ #

    @property
    def phase(self) -> Phase:
        return self.state.phase

    @property
    def finished(self) -> bool:
        return self.state.finished

    def our_options(self) -> tuple[Action, ...]:
        return legal_actions(self.state, US)

    def their_slot_of(self, species_id: str) -> int:
        """Which of their registered six this species is, by id."""
        try:
            return self.their_six.index(species_id)
        except ValueError:
            raise MirrorError(
                f"{species_id} is not one of the six they registered") from None

    def advise(self, search) -> object:
        """What the search would play here. Returns its ``Result``."""
        return search.choose(self.state, US, self.cursor)

    # -- the person at the keyboard ----------------------------------------- #

    def their_lead(self, species_id: str) -> None:
        """Who they led with, which is all preview tells us about their three.

        The other two are placeholders -- unrevealed slots that ``report`` will
        swap for whatever actually walks out. The search does not care: it
        resamples the unrevealed ones from their registered six anyway.
        """
        if self.state.phase is not Phase.TEAM_PREVIEW:
            raise MirrorError("their lead is only chosen at team preview")
        lead = self.their_slot_of(species_id)
        rest = [slot for slot in range(len(self.their_six)) if slot != lead]
        self._their_selection = (lead, *rest[:self.config.brought - 1])

    def choose_ours(self, order) -> None:
        """Our three, in the order we are bringing them."""
        self._our_selection = tuple(order)

    def open(self) -> None:
        """Submit both team-preview picks and start the battle."""
        ours = getattr(self, "_our_selection", None)
        theirs = getattr(self, "_their_selection", None)
        if ours is None or theirs is None:
            raise MirrorError("both leads have to be entered before the battle "
                              "can open")
        self._step(Action.select(*ours), Action.select(*theirs))

    def report_move(self, move_id: str) -> Action:
        """Their active used this move. Teaches it if we had not seen it."""
        slot = self._their_active_slot()
        return Action.move(self._teach(slot, move_id))

    def report_switch(self, species_id: str) -> Action:
        """They switched to this species, revealing it if it is new."""
        return Action.switch(self._reveal(species_id))

    def advance(self, ours: Action, theirs: Action) -> list:
        """Play the turn both sides committed to. Returns the engine's events."""
        return self._step(ours, theirs)

    def observe(self, side: int, hp_fraction: float | None = None,
                status: str | None = ..., position: int = 0) -> None:
        """Overwrite the active's HP and status with what the screen shows.

        **This is the correction, and it is the point.** Our damage roll is not
        the game's, our guess at their spread is not their spread, and after two
        exchanges the mirrored HP can be out by a third. Everything downstream --
        the search's leaf value most of all -- is reading these numbers.

        ``hp_fraction`` is what the bar shows, in 0..1. ``status`` defaults to
        leaving it alone; pass ``None`` explicitly to clear one.
        """
        side_state = self.state.sides[side]
        if position >= len(side_state.active):
            raise MirrorError(f"side {side} has no position {position}")
        slot = side_state.active[position]
        if slot < 0:
            raise MirrorError(f"nobody is standing in position {position}")
        if hp_fraction is not None:
            maximum = self.state.pokemon(side, slot).max_hp
            # Never round a living Pokemon down to fainted: the bar shows a
            # sliver at 1% and the engine would call the battle over.
            hp = max(1, round(maximum * hp_fraction)) if hp_fraction > 0 else 0
            side_state.hp[slot] = min(maximum, hp)
        if status is not ...:
            side_state.status[slot] = status
            if status is None:
                side_state.status_data[slot] = {}
                self.state.revealed[side].status_since.pop(slot, None)
            else:
                self.state.revealed[side].status_since.setdefault(
                    slot, self.state.turn)

    # -- keeping the placeholder honest -------------------------------------- #

    def _their_active_slot(self) -> int:
        side = self.state.sides[THEM]
        if not side.active or side.active[0] < 0:
            raise MirrorError("they have nobody on the field")
        return side.active[0]

    def _teach(self, slot: int, move_id: str) -> int:
        """Put a move on their placeholder, and say which index it landed at.

        A placeholder's four moves are a guess. When the guess is wrong the
        engine cannot step the turn at all, so the watched move replaces one we
        have *not* watched -- never one we have, because those are facts.
        """
        dex = self.config.dex
        if move_id not in dex.moves:
            raise MirrorError(f"no such move: {move_id!r}")
        party_index = self.state.sides[THEM].selection[slot]
        pokemon = self.state.parties[THEM][party_index]
        existing = [move.id for move in pokemon.moves]
        if move_id in existing:
            return existing.index(move_id)

        watched = self.state.revealed[THEM].moves_of(slot)
        spare = next((index for index, move in enumerate(existing)
                      if move not in watched), None)
        if spare is None:
            raise MirrorError(
                f"they have already shown {MAX_MOVES} moves on this Pokemon "
                f"and now a {move_id}; one of the reports must be wrong")
        moves = list(existing)
        moves[spare] = move_id
        self._rewrite(party_index, moves=tuple(moves))
        # PP is per brought slot and indexed the same way.
        self.state.sides[THEM].pp[slot][spare] = max_pp(dex.moves[move_id].pp)
        return spare

    def _reveal(self, species_id: str) -> int:
        """Their brought-party slot for this species, inventing one if needed.

        Preview says which six they registered, never which three they bring,
        so two of their three start as arbitrary picks. The first time one of
        those is contradicted by a Pokemon actually walking out, the unrevealed
        placeholder is swapped for the real species. Only slots that have never
        been on the field may be swapped -- one that has is a fact.
        """
        side = self.state.sides[THEM]
        registered = self.their_slot_of(species_id)
        for slot, party_index in enumerate(side.selection):
            if party_index == registered:
                return slot

        seen = self.state.revealed[THEM].species
        spare = next((slot for slot in range(len(side.selection))
                      if slot not in seen), None)
        if spare is None:
            raise MirrorError(
                f"they have already shown {len(side.selection)} Pokemon and "
                f"now a {species_id}; one of the reports must be wrong")
        selection = list(side.selection)
        selection[spare] = registered
        side.selection = tuple(selection)
        # The slot was never on the field, so its per-slot arrays are still at
        # their opening values -- except HP, which is sized to the old species.
        side.hp[spare] = self.state.pokemon(THEM, spare).max_hp
        side.pp[spare] = [max_pp(move.pp)
                          for move in self.state.pokemon(THEM, spare).moves]
        return spare

    def _rewrite(self, party_index: int, **changes) -> None:
        """Replace one of their registered sets, keeping everything else."""
        old = self.state.parties[THEM][party_index]
        built = compile_team(self.config.dex,
                             (_replace_set(old.set, **changes),))[0]
        parties = list(self.state.parties)
        theirs = list(parties[THEM])
        theirs[party_index] = built
        parties[THEM] = tuple(theirs)
        self.state.parties = tuple(parties)

    def _step(self, ours: Action, theirs: Action) -> list:
        self.state, events = step(self.state, ours, theirs)
        return events


def _replace_set(pokemon_set: PokemonSet, **changes) -> PokemonSet:
    from dataclasses import replace

    return replace(pokemon_set, **changes)


def _placeholder(dex: Dex, species_id: str, cursor: RngCursor) -> PokemonSet:
    """A plausible set for a species we know nothing else about.

    From the ranker pool when it has one, because that is a set a person built
    and its damage will be in the right neighbourhood. Otherwise a legal
    fallback: the species' first ability, whatever it can learn, no item.
    """
    from pkcm.engine.legality import learnable_moves, registrable_abilities
    from pkcm.envs.belief import sets_by_species

    pool = sets_by_species().get(species_id)
    if pool:
        return pool[cursor.between(0, len(pool) - 1)]

    species = dex.species[species_id]
    learnable = sorted(learnable_moves(dex, species_id))
    moves = tuple(learnable[:MAX_MOVES]) or ("struggle",)
    abilities = registrable_abilities(species)
    return PokemonSet(
        species=species_id,
        ability=abilities[0] if abilities else "__none__",
        moves=moves,
        item=None,
        nature="serious",
        sp=(11, 11, 11, 11, 11, 11),
    )
