"""How good is this position, from one player's side?

Three answers, and the search uses whichever it is told to.

**Heuristic.** Count what is left. Cheap enough to call at every leaf, and it
already knows the thing that decides most games: a Pokemon still standing is
worth more than the HP it is standing on, because it is a move every turn for
the rest of the battle.

**Pressure.** Material plus *who is about to knock out whom*. Counting material
is blind at the depth this search actually reaches: measured, the tree gets 2.8
turns deep in singles and 1.8 in doubles, and almost nothing has fainted by
then. "My Garchomp kills their Charizard next turn" and "their Charizard kills
my Garchomp" score identically under a material count, which is why the root Q
spread came out at 0.037 and the policy targets came out near uniform.

The arithmetic is already in this repository -- ``envs.analysis`` computes
damage brackets and knockout chances -- but it takes an ``Observation``,
because it is the *agent's* calculator and must not see through the fog. Inside
the search we are on a determinization that has already guessed everything
hidden, so reading the state here is legitimate, and it has to be much cheaper
than ``assess``: this runs at every leaf.

**Rollout.** Play the position out with a fast policy and take the result. Less
biased and far more expensive: a battle runs about thirty turns, so a rollout
from move one costs as much as thirty leaf evaluations.

Both return a number in ``[-1, 1]`` from ``player``'s point of view, matching
what the environment pays at the end, so the search optimises the same thing the
learner will.
"""

from __future__ import annotations

from pkcm.engine.moves import damage_base, damage_from_base
from pkcm.engine.state import BOOST_INDEX, BattleState, Phase
from pkcm.engine.mutate import stage_multiplier

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


#: The damage roll pressure is evaluated at. The median, not the maximum:
#: an evaluation that assumes every roll is high reads a 15/16 kill and a
#: certain kill as the same thing.
MEDIAN_ROLL = 92


def _boosted(state: BattleState, side: int, slot: int, index: int) -> float:
    """A raw stat with its stage applied.

    ``mutate.effective_stat`` is the full answer and needs a ``Context`` to run
    the ability and item hooks. There is no context at a leaf, and building one
    to price a guess would cost more than the guess is worth.
    """
    raw = state.stats(side, slot)[index]
    boosts = state.sides[side].boosts
    if slot >= len(boosts):
        return float(raw)
    # ``stats`` is indexed with HP first; ``BOOST_INDEX`` has no HP, because
    # nothing boosts it. Hence the shift.
    stage = boosts[slot][BOOST_INDEX[("atk", "def", "spa", "spd", "spe")[index - 1]]]
    return raw * stage_multiplier(stage)


def _threat(state: BattleState, attacker, defender) -> float:
    """Share of the defender's **remaining** health the attacker's best move takes.

    Remaining, not maximum: a Pokemon at 12% is a different problem from the
    same Pokemon at full, and the whole point of this function is to notice
    that one of them is about to faint.
    """
    chart = state.config.dex.type_chart
    attack_side, attack_slot = attacker
    defend_side, defend_slot = defender

    living = state.sides[defend_side].hp[defend_slot]
    if living <= 0:
        return 0.0

    defender_types = state.types(defend_side, defend_slot)
    attacker_types = state.types(attack_side, attack_slot)
    best = 0.0
    for move in state.moves(attack_side, attack_slot):
        power = move.base_power
        if not power:
            continue
        effectiveness = chart.multiplier(move.type, defender_types)
        if not effectiveness:
            continue
        physical = move.category == "Physical"
        offence = _boosted(state, attack_side, attack_slot, 1 if physical else 3)
        defence = _boosted(state, defend_side, defend_slot, 2 if physical else 4)
        base = damage_base(power=power, attack=int(offence), defense=int(defence))
        damage = damage_from_base(
            base, MEDIAN_ROLL,
            stab=move.type in attacker_types,
            effectiveness=effectiveness)
        best = max(best, damage / living)
    return min(1.0, best)


def pressure(state: BattleState, player: int, weight: float | None = None) -> float:
    """Material, plus who is about to knock out whom, in ``[-1, 1]``.

    The second term is what the material count cannot see at two turns' depth.
    Speed breaks the tie: between two Pokemon that each finish the other, the
    faster one is the one that gets to.
    """
    if state.phase is Phase.FINISHED:
        return terminal_value(state, player)

    material = heuristic(state, player)
    mine = state.active_refs(player)
    theirs = state.active_refs(1 - player)
    if not mine or not theirs:
        return material

    def side_threat(attackers, defenders) -> float:
        scores = []
        for attacker in attackers:
            best = max((_threat(state, attacker, defender)
                        for defender in defenders), default=0.0)
            scores.append(best)
        return sum(scores) / len(scores) if scores else 0.0

    ours = side_threat(mine, theirs)
    yours = side_threat(theirs, mine)

    # Only when both sides are close to finishing something does moving first
    # decide it; otherwise speed is already priced into the threats.
    share = PRESSURE_WEIGHT if weight is None else weight
    if ours > 0.9 and yours > 0.9:
        fast = max(_boosted(state, *ref, 5) for ref in mine)
        slow = max(_boosted(state, *ref, 5) for ref in theirs)
        if fast != slow:
            edge = share * 2 * (1 if fast > slow else -1)
            return max(-1.0, min(1.0, material + edge))

    return max(-1.0, min(1.0, material + share * (ours - yours)))


#: How much "about to knock out" counts against material.
#:
#: **The first value tried was 0.5 and it was far too large.** Measured, the
#: material term's median magnitude is 0.024 in singles and 0.124 in doubles,
#: while the threat term's is 0.127 and 0.162 -- so at 0.5 this did not add to
#: material, it replaced it. That matters more than it sounds: ablating the
#: leaf evaluation entirely costs 19 points in singles and 28 in doubles, so
#: material is one of the largest measured contributors in this search, and
#: drowning it out is expensive. At 0.5 the result was 46.7% singles, 42.0%
#: doubles, the latter separable.
PRESSURE_WEIGHT = 0.05


def terminal_value(state: BattleState, player: int) -> float:
    """What the environment would pay. A draw is nothing, not half a win."""
    if state.winner is None:
        return 0.0
    return 1.0 if state.winner == player else -1.0
