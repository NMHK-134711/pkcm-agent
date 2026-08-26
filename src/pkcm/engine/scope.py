"""What the engine can actually execute right now.

Kept separate from ``legality`` on purpose (docs/DESIGN.md §1g). Legality asks
"may this set be registered in Champions?"; this asks "can our engine run it
faithfully?". They answer to different authorities and move on different
schedules -- one changes when a regulation changes, the other when we finish a
piece of work.

The subtle case this guards is the **status move with no declarative payload**.
Showdown's client data describes most effects as fields -- ``boosts``,
``status``, ``volatileStatus``, ``sideCondition`` -- and our executor reads them
directly. But moves whose effect lives in handler code we do not have (Haze,
Roar, Rest, Trick, Baton Pass) arrive here as an empty shell. Running one would
look like a move that simply did nothing, which is exactly the kind of quiet
wrongness a policy would learn to exploit. So they are named as unsupported and
the engine logs an ``unimplemented`` event if one is ever used.

The predicate itself lives in ``pkcm.engine.moves`` beside the executor whose
capability it describes; this module is where it is looked at from outside.
"""

from __future__ import annotations

from pkcm.data.dex import Dex, Move
from pkcm.engine.moves import (
    COUNTER_MOVES,
    DECLARATIVE_FIELDS,
    FORCE_SWITCH,
    MULTI_TURN,
    NO_EFFECT_DATA,
    SELF_DESTRUCT,
    SELF_SWITCH,
    SPECIAL_CASED,
    SPECIAL_DAMAGE,
    VARIABLE_POWER_REASON,
    move_support,
)

__all__ = [
    "COUNTER_MOVES", "DECLARATIVE_FIELDS", "FORCE_SWITCH", "MULTI_TURN",
    "NO_EFFECT_DATA", "SELF_DESTRUCT", "SELF_SWITCH", "SPECIAL_CASED",
    "SPECIAL_DAMAGE", "VARIABLE_POWER_REASON",
    "move_support", "is_supported", "supported_moves", "coverage",
]


def is_supported(move: Move) -> bool:
    return move_support(move) is None


def supported_moves(dex: Dex, move_ids) -> list[str]:
    return [move_id for move_id in move_ids if is_supported(dex.moves[move_id])]


def coverage(dex: Dex) -> dict[str, int]:
    """How many standard moves the engine runs, and why the rest are out."""
    counts: dict[str, int] = {}
    for move in dex.moves.values():
        if move.raw.get("isNonstandard") is not None:
            continue
        reason = move_support(move) or "supported"
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))
