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

    ``limit`` truncates. Doubles can offer a few hundred combinations and a
    search that expands all of them learns nothing about any of them; the
    caller decides how wide it can afford to look.
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
        return combinations[:limit]
    return combinations


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
