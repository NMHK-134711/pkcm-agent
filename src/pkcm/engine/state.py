"""Battle state, split by what changes.

The split is the whole point (docs/DESIGN.md §1a). A search clones states by the
thousand, so anything constant for the battle's duration is shared by reference
and only the volatile part is copied:

``BattleConfig``   constant   dex, regulation, format, limits
``parties``        constant   the compiled Pokemon; stats, moves, max PP
``SideState``      volatile   HP, PP, status, boosts, volatiles, who is active
``FieldState``     volatile   weather, terrain, room effects
``BattleState``    volatile   phase, turn, RNG, winner

``clone()`` therefore copies a handful of small lists and dicts, never a dex
entry and never a move table.

A side holds one *field position* per active Pokemon: one in singles, two in
doubles. ``SideState.active`` maps position -> party slot, so a ``Ref`` still
names a Pokemon by the slot it was brought in, not by where it happens to be
standing. That is what lets HP, PP, status, boosts and volatiles stay indexed
by party slot in both formats, and it is why doubles needed no second copy of
any of them.

Two lifetimes live inside ``SideState`` and the difference matters:

* **Slot-persistent** -- HP, PP, major status. Survives switching out.
* **Field-only** -- stat stages, volatile conditions. Wiped the moment the
  Pokemon leaves the field. Stored per slot anyway and cleared on switch-out,
  because that keeps ``clone`` a flat list copy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from pkcm.data.dex import Dex, Regulation, Stat
from pkcm.engine.actions import (
    TARGET_ALLY,
    TARGET_SELF,
    Action,
    ActionKind,
    team_selections,
)
from pkcm.engine.pokemon import BattlePokemon, Team, compile_team
from pkcm.engine.rng import Rng

#: Champions caps games by clock, not by turn. We need a hard bound anyway so a
#: pathological self-play match cannot run forever; 200 turns is far beyond any
#: real 20-minute game and is decided on remaining Pokemon when reached.
DEFAULT_TURN_LIMIT = 200

#: Stat stages cover the five battle stats plus accuracy and evasion. HP has no
#: stage, which is why this is a separate axis from ``Stat``.
BOOST_STATS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
BOOST_INDEX = {name: index for index, name in enumerate(BOOST_STATS)}
MAX_BOOST = 6
MIN_BOOST = -6

#: Major status conditions. At most one at a time, and it survives switching.
MAJOR_STATUSES = ("brn", "par", "psn", "tox", "slp", "frz")

#: Key inside an override entry naming the fields that survive switching out.
PERMANENT = "__permanent__"

#: (side, party slot). Identifies one Pokemon for the whole battle, wherever it
#: happens to be standing. ``pkcm.engine.effects`` re-exports this name.
Ref = tuple[int, int]


class Phase(IntEnum):
    TEAM_PREVIEW = 0
    BATTLE = 1
    FORCED_SWITCH = 2
    FINISHED = 3
    #: A turn stopped halfway because someone used U-turn and owes a
    #: replacement *before* the rest of the turn happens. Distinct from
    #: FORCED_SWITCH, which comes after the turn is over.
    MID_TURN_SWITCH = 4


@dataclass(frozen=True, slots=True)
class BattleConfig:
    dex: Dex
    regulation: Regulation
    battle_format: str = "singles"
    turn_limit: int = DEFAULT_TURN_LIMIT

    @property
    def registered(self) -> int:
        return self.regulation.bring_select(self.battle_format)[0]

    @property
    def brought(self) -> int:
        return self.regulation.bring_select(self.battle_format)[1]

    @property
    def active_count(self) -> int:
        """Field positions per side: 1 in singles, 2 in doubles."""
        return 2 if self.battle_format == "doubles" else 1

    @property
    def is_doubles(self) -> bool:
        return self.active_count > 1


@dataclass(slots=True)
class FieldState:
    """Conditions that belong to the battlefield rather than to either side."""

    weather: str | None = None
    weather_turns: int = 0
    terrain: str | None = None
    terrain_turns: int = 0
    #: Trick Room, Gravity, Magic Room, Wonder Room -> turns remaining.
    rooms: dict[str, int] = field(default_factory=dict)

    def clone(self) -> "FieldState":
        return FieldState(
            weather=self.weather,
            weather_turns=self.weather_turns,
            terrain=self.terrain,
            terrain_turns=self.terrain_turns,
            rooms=dict(self.rooms),
        )


@dataclass(slots=True)
class SideState:
    """One player's volatile state. Sized to the Pokemon actually brought."""

    #: Indices into the registered team, in the order they were brought.
    selection: tuple[int, ...] = ()
    hp: list[int] = field(default_factory=list)
    pp: list[list[int]] = field(default_factory=list)
    #: Field position -> index into ``selection``; -1 while a position is empty.
    #: One entry in singles, two in doubles.
    active: list[int] = field(default_factory=list)
    #: Per position: this one's occupant fainted and owes a replacement.
    must_switch: list[bool] = field(default_factory=list)

    # -- slot-persistent, survives switching --------------------------------- #
    status: list[str | None] = field(default_factory=list)
    #: Per-slot counters the status needs: sleep turns left, Toxic's stacking N.
    status_data: list[dict[str, int]] = field(default_factory=list)

    # -- field-only, cleared on switch-out ----------------------------------- #
    boosts: list[list[int]] = field(default_factory=list)
    volatiles: list[dict[str, Any]] = field(default_factory=list)

    # -- side-wide ----------------------------------------------------------- #
    #: Reflect, Light Screen, Spikes, Stealth Rock, Tailwind -> turns or layers.
    conditions: dict[str, int] = field(default_factory=dict)

    def clone(self) -> "SideState":
        return SideState(
            selection=self.selection,
            hp=self.hp.copy(),
            pp=[slot.copy() for slot in self.pp],
            active=self.active.copy(),
            must_switch=self.must_switch.copy(),
            status=self.status.copy(),
            status_data=[data.copy() for data in self.status_data],
            boosts=[slot.copy() for slot in self.boosts],
            volatiles=[dict(slot) for slot in self.volatiles],
            conditions=dict(self.conditions),
        )

    # -- queries ------------------------------------------------------------- #

    def is_fainted(self, slot: int) -> bool:
        return self.hp[slot] <= 0

    def living_slots(self) -> list[int]:
        return [slot for slot in range(len(self.hp)) if self.hp[slot] > 0]

    def has_lost(self) -> bool:
        return bool(self.hp) and not self.living_slots()

    def active_slots(self) -> list[int]:
        """Party slots standing on the field right now, fainted ones dropped.

        A fainted Pokemon stays in its position until a replacement is sent,
        so that ``must_switch`` knows which position it owes. Everything that
        asks "who is out there" wants it gone, which is what this is for.
        """
        return [slot for slot in self.active if slot >= 0 and self.hp[slot] > 0]

    def position_of(self, slot: int) -> int | None:
        """Where a party slot is standing, or ``None`` if it is on the bench."""
        for position, occupant in enumerate(self.active):
            if occupant == slot:
                return position
        return None

    def owes_switch(self) -> bool:
        return any(self.must_switch)

    def boost(self, slot: int, stat: str) -> int:
        return self.boosts[slot][BOOST_INDEX[stat]]

    def has_volatile(self, slot: int, name: str) -> bool:
        return name in self.volatiles[slot]

    def clear_on_switch_out(self, slot: int) -> None:
        """Everything a Pokemon loses by leaving the field."""
        self.boosts[slot] = [0] * len(BOOST_STATS)
        self.volatiles[slot] = {}


@dataclass(slots=True)
class Revealed:
    """What the opponent has been shown about one side.

    Kept on the state rather than accumulated by a wrapper, because
    ``Observation(state, player)`` has to be a pure function of the state for
    search to determinize from it (docs/DESIGN.md §1c). A wrapper that folds the
    log works for a training loop and is useless to a tree search, which arrives
    at a node without having watched how it got there.

    Every revelation is already an event, so this is folded in ``Context.emit``
    -- one place, and one that cannot be incomplete for an event that exists.
    """

    #: Party slots the opponent has seen on the field.
    species: set[int] = field(default_factory=set)
    #: Party slot -> move ids the opponent has watched it use.
    moves: dict[int, set[str]] = field(default_factory=dict)
    #: Slots whose held item has been shown (used, knocked off, Frisked).
    items: set[int] = field(default_factory=set)
    #: Slots whose ability has announced itself.
    abilities: set[int] = field(default_factory=set)

    def clone(self) -> "Revealed":
        return Revealed(
            species=set(self.species),
            moves={slot: set(moves) for slot, moves in self.moves.items()},
            items=set(self.items),
            abilities=set(self.abilities),
        )

    def saw_move(self, slot: int, move_id: str) -> None:
        self.moves.setdefault(slot, set()).add(move_id)

    def moves_of(self, slot: int) -> frozenset[str]:
        return frozenset(self.moves.get(slot, ()))


@dataclass(slots=True)
class BattleState:
    config: BattleConfig
    #: The six compiled Pokemon per side. Constant; shared across clones.
    parties: tuple[tuple[BattlePokemon, ...], tuple[BattlePokemon, ...]]
    sides: tuple[SideState, SideState]
    rng: Rng
    #: Champions allows one Mega Evolution per player per battle. Declared
    #: before ``field`` because that attribute shadows ``dataclasses.field``
    #: for everything after it.
    mega_used: list[bool] = field(default_factory=lambda: [False, False])
    #: What each side has shown the other, indexed by the side being *observed*.
    #: Declared up here for the same reason ``mega_used`` is.
    revealed: tuple[Revealed, Revealed] = field(
        default_factory=lambda: (Revealed(), Revealed()))
    #: The turn currently being resolved: the actions both sides chose, and who
    #: has yet to act. Kept on the state so a turn interrupted by a self-switch
    #: can be picked up again on the next ``step``.
    turn_actions: tuple = ()
    turn_queue: list[int] = field(default_factory=list)
    field: FieldState = field(default_factory=FieldState)
    phase: Phase = Phase.TEAM_PREVIEW
    turn: int = 0
    winner: int | None = None
    #: Per-slot overrides of what a Pokemon *is*, as opposed to how it is doing.
    #:
    #: Most of them last only while the Pokemon is on the field: Protean's
    #: retype, Transform, Trace's borrowed ability, Stance Change's forme all
    #: revert when it leaves. A few do not -- Mega Evolution and a busted
    #: Disguise are done for the rest of the battle. The ones that stay are
    #: named in the entry's ``__permanent__`` key.
    #: Mega Evolution rewrites species/ability/stats; Transform rewrites nearly
    #: everything but HP. Empty dict means "as registered". Keeping this one
    #: structure rather than a field per mechanic is what stops Transform from
    #: needing a second pass through the whole engine later.
    overrides: tuple[list[dict[str, Any]], list[dict[str, Any]]] = ((), ())  # type: ignore[assignment]

    def clone(self) -> "BattleState":
        return BattleState(
            config=self.config,
            parties=self.parties,
            sides=(self.sides[0].clone(), self.sides[1].clone()),
            rng=self.rng,
            field=self.field.clone(),
            phase=self.phase,
            turn=self.turn,
            winner=self.winner,
            mega_used=list(self.mega_used),
            turn_actions=self.turn_actions,
            turn_queue=list(self.turn_queue),
            revealed=(self.revealed[0].clone(), self.revealed[1].clone()),
            overrides=(
                [dict(slot) for slot in self.overrides[0]],
                [dict(slot) for slot in self.overrides[1]],
            ),
        )

    # -- lookups ----------------------------------------------------------- #

    def pokemon(self, side: int, slot: int) -> BattlePokemon:
        """The Pokemon in a brought-party slot, as registered."""
        return self.parties[side][self.sides[side].selection[slot]]

    def active_pokemon(self, side: int, position: int = 0) -> BattlePokemon:
        return self.pokemon(side, self.sides[side].active[position])

    # -- who is on the field ------------------------------------------------ #

    def active_refs(self, side: int) -> list[Ref]:
        """Every Pokemon this side has standing, in field-position order."""
        return [(side, slot) for slot in self.sides[side].active_slots()]

    def ref_at(self, side: int, position: int) -> Ref | None:
        """Whoever occupies one field position, or ``None`` if it is empty."""
        occupants = self.sides[side].active
        if position >= len(occupants):
            return None
        slot = occupants[position]
        if slot < 0 or self.sides[side].hp[slot] <= 0:
            return None
        return (side, slot)

    def foes(self, ref: Ref) -> list[Ref]:
        """The opposing Pokemon on the field. Both of them, in doubles."""
        return self.active_refs(1 - ref[0])

    def ally(self, ref: Ref) -> Ref | None:
        """The partner standing beside this one. Always ``None`` in singles."""
        for other in self.active_refs(ref[0]):
            if other != ref:
                return other
        return None

    def allies_and_self(self, ref: Ref) -> list[Ref]:
        return self.active_refs(ref[0])

    def everyone(self) -> list[Ref]:
        return self.active_refs(0) + self.active_refs(1)

    def override(self, side: int, slot: int) -> dict[str, Any]:
        return self.overrides[side][slot] if self.overrides[side] else {}

    def set_override(self, side: int, slot: int, key: str, value: Any,
                     permanent: bool = False) -> None:
        entry = self.overrides[side][slot]
        entry[key] = value
        if permanent:
            entry[PERMANENT] = frozenset(entry.get(PERMANENT, ())) | {key}

    def clear_temporary_overrides(self, side: int, slot: int) -> None:
        """Everything a Pokemon stops being the moment it leaves the field."""
        entry = self.overrides[side][slot]
        permanent = frozenset(entry.get(PERMANENT, ()))
        kept = {key: value for key, value in entry.items() if key in permanent}
        if permanent:
            kept[PERMANENT] = permanent
        self.overrides[side][slot] = kept

    def species_id(self, side: int, slot: int) -> str:
        """Current forme, which Mega Evolution and Transform can change."""
        return self.override(side, slot).get("species") or self.pokemon(side, slot).species.id

    def species_name(self, side: int, slot: int) -> str:
        """Display name of the current forme, for logs and renderers."""
        return self.config.dex.species[self.species_id(side, slot)].name

    def ability_id(self, side: int, slot: int) -> str:
        return self.override(side, slot).get("ability") or self.pokemon(side, slot).ability

    def item_id(self, side: int, slot: int) -> str | None:
        override = self.override(side, slot)
        if "item" in override:
            return override["item"]
        return self.pokemon(side, slot).item

    def mega_target(self, side: int, slot: int) -> str | None:
        """The Mega forme this Pokemon could become right now, if any.

        Showdown's ``canMegaEvo`` is one lookup: does the held stone list this
        Pokemon's base species? Everything else -- already Mega, already spent,
        wrong holder -- falls out of that.
        """
        if self.mega_used[side]:
            return None
        species = self.config.dex.species[self.species_id(side, slot)]
        if species.is_mega:
            return None
        return self.config.dex.mega_evolution(species.id, self.item_id(side, slot))

    def can_mega_evolve(self, side: int, slot: int) -> bool:
        return self.mega_target(side, slot) is not None

    def gender(self, side: int, slot: int) -> str | None:
        return self.pokemon(side, slot).gender

    def types(self, side: int, slot: int) -> tuple[str, ...]:
        override = self.override(side, slot)
        types = (override["types"] if "types" in override
                 else self.config.dex.species[self.species_id(side, slot)].types)
        # Roost sheds the Flying type for the turn it is used.
        if slot < len(self.sides[side].volatiles) and \
                "roost" in self.sides[side].volatiles[slot]:
            grounded = tuple(t for t in types if t != "flying")
            return grounded or ("normal",)
        return types

    def stats(self, side: int, slot: int) -> tuple[int, ...]:
        """Raw stats before stat stages. HP is the registered Pokemon's own."""
        override = self.override(side, slot)
        if "stats" in override:
            return override["stats"]
        return self.pokemon(side, slot).stats

    def moves(self, side: int, slot: int) -> tuple:
        override = self.override(side, slot)
        if "moves" in override:
            return tuple(self.config.dex.moves[m] for m in override["moves"])
        return self.pokemon(side, slot).moves

    def active_hp(self, side: int, position: int = 0) -> int:
        return self.sides[side].hp[self.sides[side].active[position]]

    def speed(self, side: int, position: int = 0) -> int:
        """Raw Speed. Stage multipliers and Speed-modifying effects live in
        ``pkcm.engine.effects``; this is the unmodified number."""
        return self.stats(side, self.sides[side].active[position])[Stat.SPE]

    @property
    def finished(self) -> bool:
        return self.phase is Phase.FINISHED


def new_battle(
    config: BattleConfig,
    teams: tuple[Team, Team],
    seed: int = 0,
) -> BattleState:
    """A battle sitting at team preview. Teams are assumed already validated."""
    return BattleState(
        config=config,
        parties=(compile_team(config.dex, teams[0]), compile_team(config.dex, teams[1])),
        sides=(SideState(), SideState()),
        rng=Rng.from_seed(seed),
        overrides=([], []),
    )


def imprisoned_moves(state: BattleState, player: int) -> frozenset[str]:
    """Move ids the opposing Imprison has sealed away.

    Imprison sits on the *opponent*, so no hook gathered on the mover can see
    it -- same shape as Mold Breaker, and answered the same way: engine-side.
    In doubles either opponent can be the one imprisoning, and the seals stack.
    """
    sealed: set[str] = set()
    for foe in state.active_refs(1 - player):
        if state.sides[foe[0]].has_volatile(foe[1], "imprison"):
            sealed.update(move.id for move in state.moves(*foe))
    return frozenset(sealed)


def uproar_in_progress(state: BattleState) -> bool:
    """Nobody sleeps while an Uproar is going, anywhere on the field."""
    return any(
        state.sides[ref[0]].has_volatile(ref[1], "uproar")
        for ref in state.everyone()
    )


#: Move targets the player picks from. Everything else -- spread moves, field
#: moves, ``self``, ``randomNormal`` -- has exactly one answer, so it produces
#: one action and the ``target`` field is ignored.
CHOOSES_A_TARGET = frozenset({"normal", "any", "adjacentFoe",
                              "adjacentAlly", "adjacentAllyOrSelf"})


def move_targets(state: BattleState, ref: Ref, move) -> list[int]:
    """The target codes this move may legally be aimed at, from ``ref``.

    Returns a single ``[0]`` whenever there is nothing to choose, which keeps
    the singles action space exactly what it was.
    """
    kind = move.target
    if kind not in CHOOSES_A_TARGET or not state.config.is_doubles:
        return [0]

    codes: list[int] = []
    if kind in ("normal", "any", "adjacentFoe"):
        codes.extend(position for position, _ in enumerate(state.sides[1 - ref[0]].active)
                     if state.ref_at(1 - ref[0], position) is not None)
    # A "normal" move may be aimed at your own partner. Rarely what you want,
    # and occasionally exactly what you want -- setting off its Weakness Policy,
    # or breaking its Substitute. Champions allows it, so the mask does.
    if kind in ("normal", "any", "adjacentAlly", "adjacentAllyOrSelf"):
        if state.ally(ref) is not None:
            codes.append(TARGET_ALLY)
    if kind == "adjacentAllyOrSelf":
        codes.append(TARGET_SELF)
    return codes or [0]


def resolve_target_code(state: BattleState, ref: Ref, code: int) -> Ref | None:
    """Turn a target code back into whoever is standing there."""
    if code == TARGET_SELF:
        return ref
    if code == TARGET_ALLY:
        return state.ally(ref)
    return state.ref_at(1 - ref[0], code)


def legal_actions(state: BattleState, player: int, position: int = 0) -> tuple[Action, ...]:
    """Everything ``player`` may legally submit for one field position.

    The engine validates against this and the PettingZoo adapter builds its
    action mask from it, so the two cannot drift apart. In singles there is one
    position and this reads exactly as it always did.
    """
    if state.phase is Phase.FINISHED:
        return (Action.PASS,)

    if state.phase is Phase.TEAM_PREVIEW:
        if position > 0:
            return (Action.PASS,)  # one selection per player, not per position
        return team_selections(state.config.registered, state.config.brought)

    side = state.sides[player]
    if position >= len(side.active):
        return (Action.PASS,)
    slot = side.active[position]

    if state.phase in (Phase.FORCED_SWITCH, Phase.MID_TURN_SWITCH):
        if not side.must_switch[position]:
            return (Action.PASS,)
        replacements = _bench(state, player)
        # The bench can empty out between the mark and the choice, when the
        # other position took the last one.
        return tuple(Action.switch(bench) for bench in replacements) or (Action.PASS,)

    # An empty position owes a replacement, not a move. Reached when the rest of
    # the side is still choosing this turn.
    if slot < 0 or side.is_fainted(slot):
        return (Action.PASS,)

    ref: Ref = (player, slot)

    if side.has_volatile(slot, "mustrecharge"):
        return (Action.PASS,)

    locked = side.volatiles[slot].get("twoturn") or side.volatiles[slot].get("lockedmove")
    if locked:
        # A lock started by a *called* move -- Sleep Talk into Fly, Dancer into
        # Petal Dance -- has no index of its own, because the Pokemon never
        # chose the move from its own list. Fall back to the id, and if that is
        # not in the list either, the lock cannot be honoured and the turn is
        # free.
        index = locked.get("move")
        if index is None and locked.get("id"):
            index = next((i for i, move in enumerate(state.moves(player, slot))
                          if move.id == locked["id"]), None)
        if index is not None and index < len(side.pp[slot]):
            return _aimed(state, ref, index, locked.get("target", 0))

    # Uproar keeps going by itself for three turns (Showdown's onLockMove).
    # Matched by move id rather than by a stored index, because the volatile is
    # created from the move's own ``self`` payload, which never sees the index.
    if side.has_volatile(slot, "uproar"):
        for index, move in enumerate(state.moves(player, slot)):
            if move.id == "uproar" and side.pp[slot][index] > 0:
                return _aimed(state, ref, index)

    # Normal turn: any move with PP left, plus any living benched Pokemon.
    volatiles = side.volatiles[slot]
    disabled = volatiles.get("disabled", {}).get("move")
    # A Choice item locks the holder into the first move it picks, until it
    # leaves the field. Enforced here so the search and the environment mask
    # see the same thing the engine does.
    choice_locked = volatiles.get("choicelock", {}).get("move")
    sealed = imprisoned_moves(state, player)
    known = state.moves(player, slot)

    usable = [
        index
        for index, pp in enumerate(side.pp[slot])
        if pp > 0 and index != disabled and (choice_locked is None or index == choice_locked)
        and not (index < len(known) and known[index].id in sealed)
    ]

    actions: list[Action] = []
    for index in usable:
        actions.extend(_aimed(state, ref, index))
    if state.can_mega_evolve(player, slot):
        for index in usable:
            actions.extend(_aimed(state, ref, index, mega=True))
    if not actions:
        actions.append(Action.struggle())

    trapped = side.has_volatile(slot, "trapped") and state.item_id(player, slot) != "shedshell"
    if not trapped:
        actions.extend(Action.switch(bench) for bench in _bench(state, player))
    return tuple(actions)


def _bench(state: BattleState, player: int) -> list[int]:
    """Living Pokemon not currently standing anywhere.

    In doubles both positions see the same bench, so two positions switching at
    once can pick the same Pokemon. ``battle.step`` refuses that pair; a mask
    per position cannot express a constraint that spans positions.
    """
    side = state.sides[player]
    standing = set(side.active)
    return [slot for slot in side.living_slots() if slot not in standing]


def _aimed(state: BattleState, ref: Ref, index: int, forced_target: int | None = None,
           mega: bool = False) -> tuple[Action, ...]:
    """One action per legal target for this move."""
    moves = state.moves(*ref)
    if index >= len(moves):
        return (Action.move(index, mega=mega),)
    if forced_target is not None:
        return (Action.move(index, mega=mega, target=forced_target),)
    return tuple(
        Action.move(index, mega=mega, target=code)
        for code in move_targets(state, ref, moves[index])
    )


def is_legal(state: BattleState, player: int, action: Action) -> bool:
    return action in legal_actions(state, player)


def action_is_switch(action: Action) -> bool:
    return action.kind is ActionKind.SWITCH
