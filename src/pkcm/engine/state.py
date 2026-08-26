"""Battle state, split by what changes.

The split is the whole point (docs/DESIGN.md §1a). A search clones states by the
thousand, so anything constant for the battle's duration is shared by reference
and only the volatile part is copied:

``BattleConfig``   constant   dex, regulation, format, limits
``parties``        constant   the compiled Pokemon; stats, moves, max PP
``SideState``      volatile   HP, PP, who is active
``BattleState``    volatile   phase, turn, RNG, winner

``clone()`` therefore copies a handful of small lists and an integer, never a
dex entry and never a move table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from pkcm.data.dex import Dex, Regulation, Stat
from pkcm.engine.actions import Action, ActionKind, team_selections
from pkcm.engine.pokemon import BattlePokemon, Team, compile_team
from pkcm.engine.rng import Rng

#: Champions caps games by clock, not by turn. We need a hard bound anyway so a
#: pathological self-play match cannot run forever; 200 turns is far beyond any
#: real 20-minute game and is decided on remaining Pokemon when reached.
DEFAULT_TURN_LIMIT = 200


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

    def clone(self) -> "SideState":
        return SideState(
            selection=self.selection,
            hp=self.hp.copy(),
            pp=[slot.copy() for slot in self.pp],
            active=self.active,
            must_switch=self.must_switch,
        )

    def is_fainted(self, slot: int) -> bool:
        return self.hp[slot] <= 0

    def living_slots(self) -> list[int]:
        return [slot for slot in range(len(self.hp)) if self.hp[slot] > 0]

    def has_lost(self) -> bool:
        return bool(self.hp) and not self.living_slots()


@dataclass(slots=True)
class BattleState:
    config: BattleConfig
    #: The six compiled Pokemon per side. Constant; shared across clones.
    parties: tuple[tuple[BattlePokemon, ...], tuple[BattlePokemon, ...]]
    sides: tuple[SideState, SideState]
    rng: Rng
    phase: Phase = Phase.TEAM_PREVIEW
    turn: int = 0
    winner: int | None = None

    def clone(self) -> "BattleState":
        return BattleState(
            config=self.config,
            parties=self.parties,
            sides=(self.sides[0].clone(), self.sides[1].clone()),
            rng=self.rng,
            phase=self.phase,
            turn=self.turn,
            winner=self.winner,
        )

    # -- lookups ----------------------------------------------------------- #

    def pokemon(self, side: int, slot: int) -> BattlePokemon:
        """The Pokemon in a brought-party slot."""
        return self.parties[side][self.sides[side].selection[slot]]

    def active_pokemon(self, side: int) -> BattlePokemon:
        return self.pokemon(side, self.sides[side].active)

    def active_hp(self, side: int) -> int:
        return self.sides[side].hp[self.sides[side].active]

    def speed(self, side: int) -> int:
        return self.active_pokemon(side).stats[Stat.SPE]

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
    actions = [
        Action.move(index)
        for index, pp in enumerate(side.pp[side.active])
        if pp > 0
    ]
    if not actions:
        actions.append(Action.struggle())
    actions.extend(
        Action.switch(slot) for slot in side.living_slots() if slot != side.active
    )
    return tuple(actions)


def is_legal(state: BattleState, player: int, action: Action) -> bool:
    return action in legal_actions(state, player)


def action_is_switch(action: Action) -> bool:
    return action.kind is ActionKind.SWITCH
