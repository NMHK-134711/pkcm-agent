"""What a player may submit, and when.

Champions is a simultaneous-move game, so both sides submit an action and the
engine resolves them together. Three phases ask for three different things:

``TEAM_PREVIEW``   an ordered selection of the Pokemon to bring; index 0 leads.
``BATTLE``         a move or a switch.
``FORCED_SWITCH``  a switch, from whichever side just lost its active Pokemon.
                   The other side submits ``PASS``.

In doubles each side decides once per *field position*, so a player submits a
tuple of actions rather than one -- see ``pkcm.engine.battle.step``. A move
action also carries a ``target``, because "the opponent" stops being a single
Pokemon the moment there are two of them.

``legal_actions`` is the single source of truth for both the engine's validation
and the environment's action mask, so the two can never disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from itertools import permutations
from typing import ClassVar


#: Where a move is aimed. Foe field positions are numbered from zero, which
#: makes ``target=0`` mean "the other side's first slot" in both formats -- so a
#: singles action is a doubles action that never needed the field.
TARGET_ALLY = -1
TARGET_SELF = -2


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
    #: Which Pokemon a move is aimed at: a foe's field position, ``TARGET_ALLY``
    #: or ``TARGET_SELF``. Ignored by moves that do not choose (spread moves,
    #: field moves, and everything in singles, where there is one answer).
    target: int = 0

    @staticmethod
    def move(index: int, mega: bool = False, target: int = 0) -> "Action":
        return Action(ActionKind.MOVE, index, mega=mega, target=target)

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
        if self.kind is ActionKind.MOVE and self.target != 0:
            aim = {TARGET_ALLY: "ally", TARGET_SELF: "self"}.get(self.target, self.target)
            return f"{prefix}move({self.index}->{aim})"
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
