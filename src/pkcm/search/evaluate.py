"""How good is this position, from one player's side?

Two answers, and the search uses whichever it is told to.

**Heuristic.** Count what is left. Cheap enough to call at every leaf, and it
already knows the thing that decides most games: a Pokemon still standing is
worth more than the HP it is standing on, because it is a move every turn for
the rest of the battle.

**Rollout.** Play the position out with a fast policy and take the result. Less
biased and far more expensive: a battle runs about thirty turns, so a rollout
from move one costs as much as thirty leaf evaluations.

Both return a number in ``[-1, 1]`` from ``player``'s point of view, matching
what the environment pays at the end, so the search optimises the same thing the
learner will.
"""

from __future__ import annotations

from pkcm.engine.state import BattleState, Phase

#: A Pokemon still on its feet counts for this much more than the HP it has
#: left. Without it the search happily trades its last Pokemon's health for
#: chip damage, because the HP arithmetic comes out even.
LIVING_WEIGHT = 3.0


def heuristic(state: BattleState, player: int) -> float:
    """Material, from ``player``'s side, in ``[-1, 1]``.

    Deliberately blunt. A sharper evaluation is a place to put a learned value
    network later; this exists so the search has something honest to lean on
    before there is one, and so a regression in the search itself is visible
    against a fixed yardstick.
    """
    if state.phase is Phase.FINISHED:
        return terminal_value(state, player)

    scores = []
    for side_index in (player, 1 - player):
        side = state.sides[side_index]
        if not side.hp:
            scores.append(0.0)
            continue
        living = len(side.living_slots())
        health = sum(
            side.hp[slot] / state.pokemon(side_index, slot).max_hp
            for slot in range(len(side.hp))
        )
        scores.append(living * LIVING_WEIGHT + health)

    total = scores[0] + scores[1]
    if total <= 0:
        return 0.0
    return (scores[0] - scores[1]) / total


def terminal_value(state: BattleState, player: int) -> float:
    """What the environment would pay. A draw is nothing, not half a win."""
    if state.winner is None:
        return 0.0
    return 1.0 if state.winner == player else -1.0
