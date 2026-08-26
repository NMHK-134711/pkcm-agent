"""What the engine can actually execute right now.

Kept separate from ``legality`` on purpose. Legality asks "may this set be
registered in Champions?"; this asks "can our engine run it faithfully?". They
answer to different authorities and move on different schedules -- one changes
when Nintendo changes a regulation, the other when we finish a milestone.

Conflating them would be a quiet disaster: a move the engine cannot execute
would become "illegal", and the day M1 lands, team legality would silently shift
underneath every trained policy.

``move_support`` returns ``None`` when the move is fully implemented, or a short
reason why not. The engine logs that reason instead of pretending the move did
nothing, and the random team generator avoids such moves so that self-play is
not quietly training on no-ops.
"""

from __future__ import annotations

from pkcm.data.dex import Dex, Move

#: Moves whose base power is computed from battle state (weight, Speed, HP,
#: remaining PP, damage taken). Showdown stores them with ``basePower: 0``.
VARIABLE_POWER = "variable base power"
STATUS_MOVE = "status move"
MULTI_HIT = "multi-hit"
TWO_TURN = "two-turn"
SELF_DESTRUCT = "self-destructing"


def move_support(move: Move) -> str | None:
    """``None`` if M0 executes this move correctly, else why it does not."""
    if move.category == "Status":
        return STATUS_MOVE
    if move.base_power == 0:
        return VARIABLE_POWER
    if move.raw.get("multihit") is not None:
        return MULTI_HIT
    if "charge" in move.flags or "recharge" in move.flags:
        return TWO_TURN
    if move.raw.get("selfdestruct") is not None:
        return SELF_DESTRUCT
    return None


def is_supported(move: Move) -> bool:
    return move_support(move) is None


def supported_moves(dex: Dex, move_ids) -> list[str]:
    return [move_id for move_id in move_ids if is_supported(dex.moves[move_id])]
