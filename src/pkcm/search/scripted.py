"""A family of one-turn agents with different temperaments.

``GreedyPolicy`` is one member of this family with its coefficients fixed:

    worth = estimate.ko_chance * 10 + estimate.percent.low / 100.0

The ``* 10`` is already a risk preference -- a very aggressive one. A chance at
a knockout is worth ten times the whole rest of the scale, so between a move
that takes 60% for certain and one that has a 60% chance to knock out, this
takes the roll every time. Turn the coefficient down and it becomes the other
kind of player. That is the whole idea here: **the same calculator, read with
different attitudes.**

What they are for:

* **Trajectory variety.** ``pkcm.train.imitate`` draws positions from greedy
  play with a quarter of the moves random. One temperament walks one corridor;
  six of them walk six, and what a network generalises to is decided by which
  positions it was shown.
* **League opponents.** A population is what keeps self-play from cycling in a
  simultaneous-move game, and a hand-built population costs nothing to run --
  no search, thousands of battles a second.
* **A ladder to measure against.** One opponent gives one number.

**They are opponents and trajectory generators, never imitation targets.**
Behaviour-cloning an agent that reads more than the observation asks a network
to predict a function of information it cannot see; it learns the conditional
mean and looks mysteriously weak. That is exactly how the pick phase failed
here -- see docs/RESUME.md. The distinction matters because ``omniscient``
below is genuinely useful and genuinely unsafe to clone.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

from pkcm.engine.actions import Action, ActionKind
from pkcm.engine.rng import Rng, RngCursor
from pkcm.engine.state import BattleState, Phase
from pkcm.search.policy import decisions_wanted, joint_actions


@dataclass(frozen=True, slots=True)
class Tactic:
    """One temperament, as coefficients over what the calculator reports."""

    name: str
    #: What a chance at a knockout is worth against the damage scale. Ten is
    #: ``GreedyPolicy``; one weighs a certain 60% against a 60% roll evenly.
    ko_weight: float = 10.0
    #: Which end of the damage bracket to believe. 0 reads the floor, 1 the
    #: ceiling. The bracket is wide because their spread is hidden, so this is
    #: optimism about the opponent's build rather than about the dice.
    roll: float = 0.0
    #: Moves are scaled by ``hit_chance ** accuracy``. Zero ignores accuracy
    #: entirely -- the player who clicks the 70% move and hopes.
    accuracy: float = 1.0
    #: What a switch is worth before its matchup is considered. Negative keeps
    #: a Pokemon in unless something else argues; positive pivots freely.
    switch_bias: float = -0.5
    #: How much the switch's matchup against what is standing there moves it.
    switch_matchup: float = 0.5
    #: How much the damage coming back this turn discourages staying in. Only
    #: shifts move-against-switch, since it is the same whichever move is used.
    threat: float = 0.0
    #: Random legal action, per mille. Coverage, not strategy.
    noise: int = 0


#: A spread of temperaments, not a ladder -- none of these is meant to be the
#: strongest. ``greedy`` reproduces ``GreedyPolicy`` so the family has a member
#: whose strength is already measured.
TACTICS: tuple[Tactic, ...] = (
    Tactic("greedy"),
    Tactic("safe", ko_weight=1.0, roll=0.0, accuracy=2.0, switch_bias=-0.3),
    Tactic("gambler", ko_weight=30.0, roll=1.0, accuracy=0.0),
    Tactic("defensive", ko_weight=4.0, switch_bias=0.1, switch_matchup=1.0,
           threat=1.0),
    Tactic("pivot", ko_weight=6.0, switch_bias=0.3, switch_matchup=1.5),
    Tactic("reckless", ko_weight=20.0, roll=1.0, accuracy=0.5,
           switch_bias=-1.5, noise=80),
)

BY_NAME = {tactic.name: tactic for tactic in TACTICS}


@dataclass
class TacticPolicy:
    """A ``Tactic``, wired up as something the engine can be handed.

    ``omniscient`` reads the opponent exactly rather than through the
    observation. As a sparring partner that is fine and interesting -- the gap
    between the two is what the hidden information is *worth*, which is a number
    worth having now that modelling the belief has turned out to be the largest
    gain this project has measured. As a source of behaviour to imitate it is
    poison; see the module docstring.
    """

    tactic: Tactic
    cursor: RngCursor
    omniscient: bool = False
    _sheet: object | None = field(default=None, repr=False)

    @staticmethod
    def seeded(name: str, seed: int, omniscient: bool = False) -> "TacticPolicy":
        return TacticPolicy(BY_NAME[name], Rng.from_seed(seed).cursor(), omniscient)

    def act(self, state: BattleState, player: int) -> tuple[Action, ...]:
        options = joint_actions(state, player)
        if not options:
            return (Action.PASS,) * decisions_wanted(state, player)
        if state.phase is not Phase.BATTLE or len(options) == 1:
            return options[self.cursor.between(0, len(options) - 1)]
        if self.tactic.noise and self.cursor.below(1000) < self.tactic.noise:
            return options[self.cursor.between(0, len(options) - 1)]

        scored = self._score(state, player)
        best = max(scored(choice) for choice in options)
        tied = [choice for choice in options if scored(choice) == best]
        return tied[self.cursor.between(0, len(tied) - 1)]

    # -- what the calculator says, read this temperament's way -------------- #

    def _score(self, state: BattleState, player: int):
        from pkcm.envs.analysis import assess
        from pkcm.envs.encoding import Vocabulary
        from pkcm.envs.reference import sheet_for

        dex = state.config.dex
        sheet = sheet_for(dex, Vocabulary.of(dex))
        observation = self._observation(state, player)
        tactic = self.tactic

        moves: dict[int, dict[int, float]] = {}
        threat: dict[int, float] = {}
        for position in range(decisions_wanted(state, player)):
            assessment = assess(observation, sheet, dex, position)
            if assessment is None:
                continue
            attacker = next((k for k in observation.own
                             if k.position == position), None)
            if attacker is None:
                continue
            order = {move_id: index for index, move_id in enumerate(attacker.moves)}
            best: dict[int, float] = {}
            for _slot, estimate in assessment.damage:
                index = order.get(estimate.move_id)
                if index is None:
                    continue
                damage = (estimate.percent.low * (1 - tactic.roll)
                          + estimate.percent.high * tactic.roll) / 100.0
                worth = (estimate.ko_chance * tactic.ko_weight + damage)
                worth *= estimate.hit_chance ** tactic.accuracy
                best[index] = max(best.get(index, 0.0), worth)
            moves[position] = best
            threat[position] = max(
                (estimate.percent.high / 100.0
                 for _slot, estimate in assessment.incoming), default=0.0)

        def switch_worth(slot: int) -> float:
            from pkcm.envs.analysis import matchup

            if not tactic.switch_matchup:
                return tactic.switch_bias
            mine = state.pokemon(player, slot)
            foes = [state.pokemon(*ref) for ref in state.active_refs(1 - player)]
            if not foes:
                return tactic.switch_bias
            edge = sum(matchup(dex, mine.moves, mine.stats, mine.species.types,
                               foe.species.id) for foe in foes) / len(foes)
            return tactic.switch_bias + tactic.switch_matchup * edge

        def total(choice: Sequence[Action]) -> float:
            worth = 0.0
            for position, action in enumerate(choice):
                if action.kind is ActionKind.MOVE:
                    worth += moves.get(position, {}).get(action.index, 0.0)
                    worth -= tactic.threat * threat.get(position, 0.0)
                elif action.kind is ActionKind.SWITCH:
                    worth += switch_worth(action.index)
            return worth

        return total

    def _observation(self, state: BattleState, player: int):
        from pkcm.envs.observation import Observation

        if not self.omniscient:
            return Observation.of(state, player)
        return _oracle(state, player)


def _oracle(state: BattleState, player: int):
    """The observation an opponent would have if nothing were hidden.

    Built by giving their side the same view we build for our own. Kept here
    rather than in ``pkcm.envs.observation`` on purpose: that module's whole
    job is to draw the line this crosses, and a function that crosses it should
    not sit where something might reach for it by accident.
    """
    from dataclasses import replace as _replace

    from pkcm.envs.observation import Observation, _own_view

    honest = Observation.of(state, player)
    foe = 1 - player
    return _replace(honest, foe=tuple(
        _own_view(state, foe, slot)
        for slot in range(len(state.sides[foe].hp))))
