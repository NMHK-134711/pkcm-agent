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
from pkcm.engine.actions import Action, ActionKind, team_selections
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


class Phase(IntEnum):
    TEAM_PREVIEW = 0
    BATTLE = 1
    FORCED_SWITCH = 2
    FINISHED = 3


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
    #: Index into ``selection``; -1 before the first switch-in.
    active: int = -1
    #: Set when this side's active fainted and owes a replacement.
    must_switch: bool = False

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
            active=self.active,
            must_switch=self.must_switch,
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

    def boost(self, slot: int, stat: str) -> int:
        return self.boosts[slot][BOOST_INDEX[stat]]

    def has_volatile(self, slot: int, name: str) -> bool:
        return name in self.volatiles[slot]

    def clear_on_switch_out(self, slot: int) -> None:
        """Everything a Pokemon loses by leaving the field."""
        self.boosts[slot] = [0] * len(BOOST_STATS)
        self.volatiles[slot] = {}


@dataclass(slots=True)
class BattleState:
    config: BattleConfig
    #: The six compiled Pokemon per side. Constant; shared across clones.
    parties: tuple[tuple[BattlePokemon, ...], tuple[BattlePokemon, ...]]
    sides: tuple[SideState, SideState]
    rng: Rng
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
            overrides=(
                [dict(slot) for slot in self.overrides[0]],
                [dict(slot) for slot in self.overrides[1]],
            ),
        )

    # -- lookups ----------------------------------------------------------- #

    def pokemon(self, side: int, slot: int) -> BattlePokemon:
        """The Pokemon in a brought-party slot, as registered."""
        return self.parties[side][self.sides[side].selection[slot]]

    def active_pokemon(self, side: int) -> BattlePokemon:
        return self.pokemon(side, self.sides[side].active)

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

    def gender(self, side: int, slot: int) -> str | None:
        return self.pokemon(side, slot).gender

    def types(self, side: int, slot: int) -> tuple[str, ...]:
        override = self.override(side, slot)
        if "types" in override:
            return override["types"]
        return self.config.dex.species[self.species_id(side, slot)].types

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

    def active_hp(self, side: int) -> int:
        return self.sides[side].hp[self.sides[side].active]

    def speed(self, side: int) -> int:
        """Raw Speed. Stage multipliers and Speed-modifying effects live in
        ``pkcm.engine.effects``; this is the unmodified number."""
        return self.stats(side, self.sides[side].active)[Stat.SPE]

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


def legal_actions(state: BattleState, player: int) -> tuple[Action, ...]:
    """Everything ``player`` may legally submit right now.

    The engine validates against this and the PettingZoo adapter builds its
    action mask from it, so the two cannot drift apart.
    """
    if state.phase is Phase.FINISHED:
        return (Action.PASS,)

    if state.phase is Phase.TEAM_PREVIEW:
        return team_selections(state.config.registered, state.config.brought)

    side = state.sides[player]

    if state.phase is Phase.FORCED_SWITCH:
        if not side.must_switch:
            return (Action.PASS,)
        return tuple(
            Action.switch(slot) for slot in side.living_slots() if slot != side.active
        )

    # Normal turn: any move with PP left, plus any living benched Pokemon.
    disabled = side.volatiles[side.active].get("disabled", {}).get("move")
    actions = [
        Action.move(index)
        for index, pp in enumerate(side.pp[side.active])
        if pp > 0 and index != disabled
    ]
    if not actions:
        actions.append(Action.struggle())
    if not side.has_volatile(side.active, "trapped"):
        actions.extend(
            Action.switch(slot) for slot in side.living_slots() if slot != side.active
        )
    return tuple(actions)


def is_legal(state: BattleState, player: int, action: Action) -> bool:
    return action in legal_actions(state, player)


def action_is_switch(action: Action) -> bool:
    return action.kind is ActionKind.SWITCH
