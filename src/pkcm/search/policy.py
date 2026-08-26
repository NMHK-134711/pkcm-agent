"""What a player does when asked. One interface, several answers.

The search needs opponents to roll out against and a baseline to be measured
by, and the environment needs something to drive it. All of them answer the same
question -- given a state and a side, what do you submit -- so all of them are
one protocol.

A policy that peeks at ``state`` is only allowed to look at what its own side
can see. ``RandomPolicy`` and ``SearchPolicy`` hold to that; ``GreedyPolicy``
uses the damage calculator, which is built on the observation and so holds to it
too. Nothing here reads the opponent's hidden fields, and a test says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

from pkcm.engine.actions import Action, ActionKind
from pkcm.engine.rng import Rng, RngCursor
from pkcm.engine.state import BattleState, Phase, legal_actions


class Policy(Protocol):
    """Anything that can be asked for a side's actions."""

    def act(self, state: BattleState, player: int) -> tuple[Action, ...]:
        """One action per field position. Team preview asks for one."""
        ...


def decisions_wanted(state: BattleState, player: int) -> int:
    return 1 if state.phase is Phase.TEAM_PREVIEW else state.config.active_count


def joint_actions(state: BattleState, player: int,
                  limit: int | None = None) -> list[tuple[Action, ...]]:
    """Every combination of per-position actions this side may submit.

    The one rule a per-position mask cannot carry is enforced here: the same
    Pokemon may not be sent to two positions. Singles collapses to a plain list
    of that position's actions.

    ``limit`` keeps the most promising ``limit`` combinations. Doubles can offer
    a few hundred and a search that expands all of them learns nothing about any
    of them.

    Which ones are kept matters more than how many. Truncating in enumeration
    order looks harmless and is not: ``legal_actions`` lists moves before
    switches, so a doubles node would drop every switch and the search would
    never consider leaving. The ordering below is crude -- power times type
    effectiveness -- but it is about the move rather than about the order the
    list happened to come in.
    """
    positions = decisions_wanted(state, player)
    combinations: list[tuple[Action, ...]] = [()]
    for position in range(positions):
        allowed = legal_actions(state, player, position)
        grown: list[tuple[Action, ...]] = []
        for prefix in combinations:
            used = {action.index for action in prefix if action.kind is ActionKind.SWITCH}
            for action in allowed:
                if action.kind is ActionKind.SWITCH and action.index in used:
                    continue
                grown.append(prefix + (action,))
        combinations = grown
        if not combinations:
            return []
    if limit is not None and len(combinations) > limit:
        combinations.sort(key=lambda choice: -_promise(state, player, choice))
        return combinations[:limit]
    return combinations


#: A switch is worth considering even when every attack scores higher, so it is
#: given a floor rather than left to lose every comparison.
SWITCH_PROMISE = 0.6


def _promise(state: BattleState, player: int, choice: tuple[Action, ...]) -> float:
    """A cheap guess at how good a choice is, for ordering only.

    Reads the state directly, which is legitimate here and nowhere else: inside
    the search this runs on a *determinization*, which has already committed to
    a guess about everything hidden. It is never used to decide anything -- only
    to pick which branches are worth the budget.
    """
    total = 0.0
    for position, action in enumerate(choice):
        if action.kind is ActionKind.SWITCH:
            total += SWITCH_PROMISE
            continue
        if action.kind is not ActionKind.MOVE:
            continue
        slot = state.sides[player].active[position] if position < len(
            state.sides[player].active) else -1
        if slot < 0:
            continue
        moves = state.moves(player, slot)
        if action.index >= len(moves):
            continue
        move = moves[action.index]
        power = move.base_power or 45      # a status move is worth a look
        best = 0.0
        for foe in state.foes((player, slot)):
            effectiveness = state.config.dex.type_chart.multiplier(
                move.type, state.types(*foe))
            best = max(best, effectiveness)
        stab = 1.5 if move.type in state.types(player, slot) else 1.0
        total += power * (best if best else 0.1) * stab / 100.0
    return total


@dataclass
class RandomPolicy:
    """Uniform over legal actions. The floor everything else is measured against."""

    cursor: RngCursor

    @staticmethod
    def seeded(seed: int) -> "RandomPolicy":
        return RandomPolicy(Rng.from_seed(seed).cursor())

    def act(self, state: BattleState, player: int) -> tuple[Action, ...]:
        options = joint_actions(state, player)
        if not options:
            return (Action.PASS,) * decisions_wanted(state, player)
        return options[self.cursor.between(0, len(options) - 1)]


@dataclass
class GreedyPolicy:
    """Pick the move with the best expected knockout, else the most damage.

    One turn deep and no more, which is exactly its point: it plays the way the
    damage calculator alone would, so beating it says the search is worth
    something beyond arithmetic. It switches only when forced.
    """

    cursor: RngCursor
    _sheet: object | None = field(default=None, repr=False)

    @staticmethod
    def seeded(seed: int) -> "GreedyPolicy":
        return GreedyPolicy(Rng.from_seed(seed).cursor())

    def act(self, state: BattleState, player: int) -> tuple[Action, ...]:
        options = joint_actions(state, player)
        if not options:
            return (Action.PASS,) * decisions_wanted(state, player)
        if state.phase is not Phase.BATTLE:
            return options[self.cursor.between(0, len(options) - 1)]

        from pkcm.envs.analysis import assess
        from pkcm.envs.encoding import Vocabulary
        from pkcm.envs.observation import Observation
        from pkcm.envs.reference import sheet_for

        dex = state.config.dex
        sheet = sheet_for(dex, Vocabulary.of(dex))
        observation = Observation.of(state, player)

        scored: dict[int, dict[int, float]] = {}
        for position in range(decisions_wanted(state, player)):
            assessment = assess(observation, sheet, dex, position)
            if assessment is None:
                continue
            attacker = next((k for k in observation.own if k.position == position), None)
            if attacker is None:
                continue
            index = {move_id: number for number, move_id in enumerate(attacker.moves)}
            best: dict[int, float] = {}
            for _slot, estimate in assessment.damage:
                move_index = index.get(estimate.move_id)
                if move_index is None:
                    continue
                # Knocking something out beats any amount of chip damage, so the
                # probability of it dominates and the damage only breaks ties.
                worth = estimate.ko_chance * 10 + estimate.percent.low / 100.0
                best[move_index] = max(best.get(move_index, 0.0), worth)
            scored[position] = best

        def value(choice: tuple[Action, ...]) -> float:
            total = 0.0
            for position, action in enumerate(choice):
                if action.kind is ActionKind.MOVE:
                    total += scored.get(position, {}).get(action.index, 0.0)
                elif action.kind is ActionKind.SWITCH:
                    total -= 0.5      # never voluntarily, but better than nothing
            return total

        best_value = max(value(choice) for choice in options)
        tied = [choice for choice in options if value(choice) == best_value]
        return tied[self.cursor.between(0, len(tied) - 1)]


@dataclass
class SearchPolicy:
    """Wraps an ``MCTS`` so a search can be dropped in wherever a policy goes."""

    search: object
    cursor: RngCursor

    def act(self, state: BattleState, player: int) -> tuple[Action, ...]:
        return self.search.choose(state, player, self.cursor).action


def play_out(state: BattleState, policies: Sequence[Policy],
             turn_limit: int | None = None) -> BattleState:
    """Run a battle to the end between two policies. Used by rollouts and evals."""
    from pkcm.engine.battle import step

    limit = turn_limit if turn_limit is not None else state.config.turn_limit
    while not state.finished and state.turn <= limit:
        choices = tuple(policies[player].act(state, player) for player in (0, 1))
        state, _ = step(state, choices[0], choices[1])
    return state


def prior_over(state: BattleState, player: int,
               options: Sequence[tuple[Action, ...]]) -> list[float]:
    """A normalised guess at which of these are worth looking at first.

    The search uses it as PUCT's prior. Without one, both sides explore
    uniformly and the budget goes on learning that a resisted move is bad --
    and worse, the *opponent* in the tree plays near-randomly for its first
    visits, so the search plans against someone weaker than it will meet.

    ``_promise`` is standing in for a policy network. When there is one, it
    replaces this function and nothing else changes.
    """
    scores = [max(0.01, _promise(state, player, choice)) for choice in options]
    total = sum(scores)
    if total <= 0:
        return [1.0 / max(1, len(options))] * len(options)
    return [score / total for score in scores]
