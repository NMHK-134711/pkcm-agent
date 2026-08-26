"""What a player may submit, and when.

Champions is a simultaneous-move game, so both sides submit an action and the
engine resolves them together. Three phases ask for three different things:

``TEAM_PREVIEW``   an ordered selection of the Pokemon to bring; index 0 leads.
``BATTLE``         a move or a switch.
``FORCED_SWITCH``  a switch, from whichever side just lost its active Pokemon.
                   The other side submits ``PASS``.

``legal_actions`` is the single source of truth for both the engine's validation
and the environment's action mask, so the two can never disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from itertools import permutations
from typing import ClassVar


class ActionKind(IntEnum):
    MOVE = 0
    SWITCH = 1
    SELECT = 2  # team preview
    PASS = 3    # nothing to decide this phase
    STRUGGLE = 4


@dataclass(frozen=True, slots=True)
class Action:
    kind: ActionKind
    index: int = 0
    selection: tuple[int, ...] = ()
    #: Mega Evolve first, then use the move. Champions allows it once a battle,
    #: so it is a property of the action rather than a separate decision -- the
    #: player commits to spending it on this turn's move.
    mega: bool = False

    @staticmethod
    def move(index: int, mega: bool = False) -> "Action":
        return Action(ActionKind.MOVE, index, mega=mega)

    @staticmethod
    def switch(slot: int) -> "Action":
        """``slot`` indexes the brought party, not the registered team."""
        return Action(ActionKind.SWITCH, slot)

    @staticmethod
    def select(*order: int) -> "Action":
        return Action(ActionKind.SELECT, selection=tuple(order))

    @staticmethod
    def struggle() -> "Action":
        return Action(ActionKind.STRUGGLE)

    #: Submitted by a side with nothing to decide this phase.
    PASS: ClassVar["Action"]

    def __str__(self) -> str:
        if self.kind is ActionKind.SELECT:
            return f"select({','.join(map(str, self.selection))})"
        if self.kind in (ActionKind.PASS, ActionKind.STRUGGLE):
            return self.kind.name.lower()
        prefix = "mega+" if self.mega else ""
        return f"{prefix}{self.kind.name.lower()}({self.index})"


Action.PASS = Action(ActionKind.PASS)


@lru_cache(maxsize=None)
def team_selections(registered: int, brought: int) -> tuple[Action, ...]:
    """Every ordered selection of ``brought`` from ``registered``.

    Order matters -- index 0 is the lead -- so this enumerates permutations,
    not combinations. Singles 6->3 gives 120, doubles 6->4 gives 360; both are
    small enough to enumerate as a flat discrete action space, and small enough
    to cache outright -- the set depends only on the format.
    """
    return tuple(Action.select(*order) for order in permutations(range(registered), brought))
