"""Determinized simultaneous-move MCTS.

The two departures from textbook MCTS are both forced by the game, and both are
in the tree rather than around it.

**Simultaneous moves.** A node does not belong to a player. It holds a set of
action statistics for *each* side and selects a pair, which is Decoupled UCT:
each player runs UCB1 over its own marginal statistics as though the other were
part of the environment. That is known not to converge to a Nash equilibrium in
general -- the marginals cannot represent a mixed strategy that needs
correlation -- but it is cheap, it is unbiased about who moves first, and it is
the standard place to start. ``select`` is one function, so replacing it with
regret matching later is a small change rather than a rewrite.

**Imperfect information.** Every iteration draws a fresh determinization from
the observation and searches *that*. The tree is shared across draws and keyed
by the sequence of joint actions, so a line that only works against one possible
opponent team gets averaged down by the draws where that team is not there.
This is IS-MCTS with a single tree, and the property worth having is that the
search cannot exploit a secret it was never told.

What it does **not** do yet: no learned value network (``evaluate.heuristic``
stands in), no progressive widening, no transposition table. Those are the
obvious next moves and none of them changes the shape here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from pkcm.engine.actions import Action
from pkcm.engine.battle import IllegalActionError, step
from pkcm.engine.rng import Rng, RngCursor
from pkcm.engine.state import BattleState, Phase
from pkcm.envs.observation import Observation, determinize
from pkcm.search.evaluate import heuristic, pressure, terminal_value
from pkcm.search.policy import (
    RandomPolicy,
    decisions_wanted,
    joint_actions,
    prior_over,
)

Choice = tuple[Action, ...]

#: What to search with when the point is to play well rather than to make
#: data cheaply. Four times the self-play budget, on the evidence in
#: ``SearchConfig.iterations`` that every doubling so far has paid.
#:
#: Not a ceiling -- the ceiling is whatever the clock allows, and no measured
#: rung has bent yet. It is the largest setting this project has actually
#: played four hundred games at.
DEPLOY_ITERATIONS = 3200

#: Added to an unvisited option's score so it outranks every visited one. The
#: value only has to exceed the range of ``mean + bias + exploration``, which is
#: bounded well below this.
UNVISITED_BONUS = 1e6


@dataclass
class MinMax:
    """The range of values one search has actually seen. MuZero's trick.

    PUCT adds a Q in the evaluation's units to an exploration bonus in units of
    nothing at all, so the two only balance if the Q happens to span roughly the
    same distance. Here it does not, and not by a little: ``heuristic`` reads
    -0.2 for a whole Pokemon lost and -0.02 for half a health bar, while the
    exploration term at eight hundred visits is worth 0.26. Measured at the
    root, the Q spread across actions was 0.054. The search was choosing almost
    entirely on exploration.

    That is not a tuning complaint, it is the reason the learning loop was dead.
    Visit counts decided by exploration are uniform by construction, 73% of the
    self-play policy targets came out within 0.15 nats of uniform, and the
    policy head settled at exactly the mean entropy of its own targets -- having
    learned everything that was there, which was nothing.

    Rescaling by the observed range makes the comparison scale-free. Whatever
    the evaluation's units, the best line this search has found sits at 1 and
    the worst at 0, so the exploration constant means the same thing whether the
    leaves are a blunt material count or a trained value head.
    """

    low: float = math.inf
    high: float = -math.inf

    def add(self, value: float) -> None:
        self.low = min(self.low, value)
        self.high = max(self.high, value)

    def scale(self, mean: float, side: int) -> float:
        """``mean`` into ``[0, 1]``, from ``side``'s point of view.

        Side 1 accumulates the negation, so its means live in the mirrored
        range. Until two distinct values have turned up there is no range to
        scale by and the number is left alone -- inventing a span would be worse
        than admitting there is not one yet.
        """
        span = self.high - self.low
        if span <= 0 or span == math.inf:
            return mean
        return (mean - self.low) / span if side == 0 else (mean + self.high) / span


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """How much thinking to do, and of what kind."""

    #: Total simulations.
    #:
    #: **This buys strength in singles and has not stopped.** Head to head on
    #: ranker teams, 400 games apiece, each rung separable:
    #:
    #:     200 -> 800    +9.7pp
    #:     800 -> 1600   +7.6pp
    #:     1600 -> 3200  +6.5pp
    #:
    #: Doubles is a different shape: 800 to 1600 came back +2.0pp and not
    #: separable, because ``max_branching`` discards half the legal moves at
    #: nearly two doubles decisions in three, and looking harder at a list the
    #: answer is missing from does not find it. See ``max_branching``.
    #:
    #: So the right number depends on what the budget is being spent for, and
    #: those are two different jobs:
    #:
    #: * **Playing.** Spend what the clock allows. A real turn timer is 45 to
    #:   90 seconds against the couple of seconds 3200 costs, so there is room
    #:   for two or three more doublings above anything measured here.
    #: * **Generating self-play data.** Time trades directly against games:
    #:   3200 simulations buys a quarter of the battles 800 does. Whether a
    #:   stronger teacher on fewer games beats a weaker one on more has not
    #:   been measured, so this stays where it was.
    #:
    #: ``DEPLOY_ITERATIONS`` is the first job's number, this default is the
    #: second's.
    iterations: int = 400
    #: How many determinizations to spread them over. One draw per
    #: ``iterations // determinizations`` simulations. More draws means a
    #: broader view of what the opponent might be holding and fewer simulations
    #: each to judge it with.
    determinizations: int = 20
    #: Turns to play out at a leaf before falling back to the heuristic. Zero
    #: uses the heuristic alone, which is much cheaper and much blunter. This is
    #: the quality dial: at twenty turns the root value went from +0.02 to +0.38
    #: on the same position, because the heuristic alone cannot see far enough
    #: to tell the lines apart. It also costs about five times as much.
    rollout_turns: int = 0
    #: A UCB1 term, *on top of* PUCT's prior term. AlphaZero has only the
    #: second, and this is small because the first mostly gets in the way.
    #:
    #: Strength does not care much: 0.1 against 0.7 came back 53.8% [46.8,
    #: 60.6] over 200 games and 0.3 against 0.7 came back 51.0% [44.1, 57.8],
    #: neither separable. The policy targets care a great deal. At 0.7 the
    #: search's visit distribution sits at 94% of uniform entropy with a 34%
    #: top action, which is a training target that says almost nothing; at 0.1
    #: it is 78% and 46%.
    #:
    #: So this is chosen for what it feeds the learner, on the evidence that it
    #: costs nothing at the board. Not on evidence that it plays better -- two
    #: runs leaned that way and neither one separated, and a lean is what this
    #: project has twice mistaken for a result.
    exploration: float = 0.1
    #: How hard the prior pulls on PUCT selection.
    #:
    #: AlphaZero's c_puct assumes Q over [-1, 1]; min-max normalisation puts
    #: ours over [0, 1], so the old 1.5 pulled about as hard as 3.0 would there
    #: -- hard enough that the search reproduced its prior instead of improving
    #: on it. Measured at 1.5: visit counts 0.037 nats sharper than the prior
    #: and the prior's own favourite kept 65% of the time. That is PUCT
    #: converging to N(a) proportional to P(a), and it leaves the learner
    #: nothing to learn.
    #:
    #: Against 1.5 over 200 games apiece: 0.75 scored 53.5% [46.6, 60.3], 0.35
    #: scored 52.0% [45.1, 58.8], 0.15 scored 49.0% [42.2, 55.9], 0.0 scored
    #: 44.5% [37.8, 51.4]. No single row separates. Four runs declining in
    #: order is still a trend, and it puts the floor around here -- loosening
    #: this far is free, loosening further starts to cost.
    #:
    #: Note what is *not* being loosened. The prior also orders the truncation
    #: in ``joint_actions``, and that job -- deciding which two dozen of a
    #: hundred and twenty options are worth looking at -- is where much of its
    #: strength lives. This dial only governs the pull inside the tree.
    #:
    #: Zero is not an option whatever it measures, because the network's policy
    #: head *is* the prior: at zero it would have no influence on the search and
    #: there would be no reason to train it.
    prior_weight: float = 0.35
    #: Deepest the tree grows. Beyond this a node is evaluated, not expanded.
    #:
    #: Twelve rather than six, and the reason is the heuristic: six turns into a
    #: thirty-turn battle almost nothing has happened, every line evaluates to
    #: roughly zero, and the root distribution comes out flat. Depth is what
    #: gives the material count something to count.
    max_depth: int = 12
    #: Cap on how many joint actions a node considers. Doubles can offer a few
    #: hundred and a node that expands all of them learns nothing about any.
    max_branching: int = 24
    #: How much a switch's matchup against what is standing there moves its
    #: prior. **Off, because it was measured and it does nothing.**
    #:
    #: The argument was good. ``_promise`` scores every switch at a flat
    #: ``SWITCH_PROMISE``, so the prior cannot rank them: decisions where it
    #: wanted a switch came back at +0.000 excess nats and 46% top-1 agreement
    #: against the pre-trained network -- a perfect copy of a uniform
    #: distribution, argmax settled by tie-break. Ranking them with the same
    #: matchup arithmetic the pick phase uses cost one function call and left
    #: the move-versus-switch balance alone.
    #:
    #: Head to head, 200 matches on ranker teams, both seatings:
    #: **51.0% [46.1, 55.9]**. Four hundred games could not tell it from the
    #: flat version. Kept because it is measured and cheap to re-test, not
    #: because it helps.
    #: What a switch is worth before its matchup is considered, against a move
    #: promise of ``power * effectiveness * stab / 100`` -- so the default 0.6
    #: is exactly a 60-power neutral move, and 40% of a same-type 100.
    #:
    #: Never measured. ``switch_matchup`` deliberately held the mean here fixed
    #: so it would only change the order among switches, which left the
    #: move-against-switch balance itself untested. hk played the agent and said
    #: it does not switch enough; measured, it switches on 18.8% of the turns
    #: where it could, 4.3 times a battle. That is not nothing, so the question
    #: is whether the level is right rather than whether it switches at all.
    switch_promise: float | None = None
    switch_matchup: float = 0.0
    #: Draw the opponent's hidden fields from the ranker pool instead of
    #: uniformly, and narrow them by what they have already shown.
    #:
    #: Head to head against the uniform version, 200 matches on ranker teams,
    #: both seatings: **68.6% [63.9, 73.0]** over 395 games. That is the largest
    #: separable gain this project has measured since the handcrafted prior
    #: itself, and it needed no network.
    #:
    #: What it fixes is that the search was planning against opponents nobody
    #: brings -- an item from all 147 the format allows, a move from the sixty
    #: that species can learn -- and that **watching a move go off narrowed
    #: nothing**, because the other three were redrawn regardless.
    #: See ``pkcm.envs.belief``.
    belief: bool = True
    #: Index children by our action alone, sampling the opponent's from its own
    #: marginal, instead of by the pair.
    #:
    #: Both sides still keep their own statistics -- that part is unchanged, and
    #: it is what makes this a simultaneous-move search rather than a
    #: turn-taking one. What changes is the *tree*: keying children by the pair
    #: makes branching |A|x|B|, so eight hundred simulations over eight options
    #: a side is thirteen visits per pair and the visit counts are mostly noise.
    #: Keying by our action alone makes it |A|, which is a hundred visits each.
    #:
    #: **Off, because it was measured and it loses.** Head to head at 200
    #: games, mirrored teams and both seatings, the search with this off beat
    #: the search with it on 59.5% [52.6, 66.1] -- an interval clear of fifty,
    #: which is as close to settled as this project gets. Planning against an
    #: opponent drawn from noisy visit counts is worse than planning against one
    #: choosing its best reply, and the narrower tree does not pay for it.
    #:
    #: It does sharpen the policy targets, which is the awkward part and the
    #: reason to leave the flag here rather than delete it. Measured at
    #: exploration 0.0, battle turns come out at 51% of uniform with it on and
    #: 70% with it off. Not through the marginal counts -- those come out of
    #: ``_select`` whatever the children are keyed by -- but through variance:
    #: a fixed opponent distribution means each of our actions is scored against
    #: the same replies, so their values are comparable rather than noisy.
    #:
    #: Sharper targets from a weaker search is a bad trade, so it stays off. The
    #: variance argument is worth keeping for whatever replaces this -- regret
    #: matching at the root would get the same effect without the loss.
    sample_opponent: bool = False
    #: Rescale Q by the range of values this search has actually met, before
    #: comparing it against the exploration bonus. See ``MinMax``.
    #:
    #: Ablated head to head it came back 54.3% [47.3, 61.0] -- not separable,
    #: so this is a claim about scale and about target sharpness, not a claim
    #: about strength, and it is on because of the first two.
    normalize_value: bool = True
    #: Leaves evaluated per network forward. One is the sequential search,
    #: exactly as before; above one, that many simulations descend the tree
    #: under **virtual loss** and their leaves go to the network as one batch.
    #:
    #: Why: a batch-1 forward costs 4.8ms and almost all of it is per-call
    #: overhead -- the same net at batch 64 is 0.17ms per state. With two
    #: forwards per expansion (each side observes the leaf its own way), the
    #: forward was ~85% of self-play. Batching pays the overhead once.
    #:
    #: The cost is not free in kind: simulations inside one batch cannot see
    #: each other's results, so the search explores slightly more diffusely
    #: than the sequential one. **Priced before this became the default**, both
    #: sides holding the same network and differing only here: 51.0% [46.1,
    #: 55.9] over 396 decided games on ranker teams. Not separable, so the
    #: diffusion costs nothing measurable and the 2.35x is free.
    #:
    #: Sixteen rather than thirty-two because sixteen is the number that was
    #: measured. Thirty-two timed 5% faster again and has never been played.
    leaf_batch: int = 16
    #: Which leaf evaluation. ``"material"`` counts what is left;
    #: ``"pressure"`` adds who is about to knock out whom; ``"blind"`` returns
    #: nothing at all except at terminals.
    #:
    #: ``"blind"`` is an ablation, not a setting to run. Every attempt to
    #: improve on the material count has come back worse -- the learned value
    #: head at 39.9%, the handcrafted threat term at 46.7% singles and 42.0%
    #: doubles -- while the two changes that did buy strength were the prior
    #: (+20.6pp) and belief (+18.6pp), neither of which is an opinion about how
    #: good a position is. So the question this answers is whether the leaf
    #: value contributes anything at all, or whether this search is a tactical
    #: filter driven by its prior with an evaluation bolted on the side.
    #:
    #: Measured, this tree reaches 2.8 turns in singles and 1.8 in doubles
    #: against a cap of twelve -- 800 simulations spread over |A|x|B| children
    #: go wide, not deep. Almost nothing has fainted that soon, so a material
    #: count returns nearly the same number down every line, and the root Q
    #: spread came out at 0.037 against an exploration term worth ten times it.
    #:
    #: **Off until measured**, like every other evaluation change here.
    evaluation: str = "material"
    #: Used when ``evaluation`` is ``"pressure"``. ``None`` takes the module
    #: default. See ``evaluate.PRESSURE_WEIGHT`` for why the first guess was
    #: ten times too large.
    pressure_weight: float | None = None


#: What a simulation pessimistically scores while it is still in flight.
#:
#: Applied to *both* sides of every node on the way down, taken back out on
#: backup. The point is only to make the next simulation in the same batch
#: look elsewhere: an in-flight line reads as one extra visit that lost, which
#: is precisely the signature of a move tried and found wanting.
VIRTUAL_LOSS = 1.0


@dataclass
class Node:
    """One position. Two players' statistics, one shared visit count."""

    options: tuple[list[Choice], list[Choice]]
    #: Normalised prior over each side's options. Sums to one per side.
    priors: tuple[list[float], list[float]] = field(default=None)  # type: ignore[assignment]
    visits: int = 0
    counts: tuple[list[int], list[int]] = field(default=None)  # type: ignore[assignment]
    totals: tuple[list[float], list[float]] = field(default=None)  # type: ignore[assignment]
    children: dict[tuple[int, int], "Node"] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.counts is None:
            self.counts = ([0] * len(self.options[0]), [0] * len(self.options[1]))
        if self.totals is None:
            self.totals = ([0.0] * len(self.options[0]), [0.0] * len(self.options[1]))
        if self.priors is None:
            self.priors = tuple(  # type: ignore[assignment]
                [1.0 / max(1, len(side))] * len(side) for side in self.options
            )

    @property
    def expanded(self) -> bool:
        return bool(self.options[0]) and bool(self.options[1])

    def mean(self, player: int, index: int) -> float:
        count = self.counts[player][index]
        return self.totals[player][index] / count if count else 0.0


@dataclass(frozen=True, slots=True)
class SearchResult:
    action: Choice
    #: Visit share per option, which is the search's policy at the root. Useful
    #: as a training target later, and as the thing to look at when it plays a
    #: move that looks wrong.
    distribution: tuple[tuple[Choice, float], ...]
    value: float
    iterations: int


class MCTS:
    def __init__(self, config: SearchConfig | None = None, evaluator=None) -> None:
        """``evaluator`` supplies the prior and the leaf value if it is given.

        Anything with ``prior(state, player, options)`` and
        ``value(state, player)`` will do -- which in practice means
        ``pkcm.train.evaluator.Evaluator``, wrapping the network. Without one
        the handcrafted versions stand in, and the tree cannot tell.
        """
        self.config = config if config is not None else SearchConfig()
        self.evaluator = evaluator

    # -- the entry point ---------------------------------------------------- #

    def choose(self, state: BattleState, player: int,
               cursor: RngCursor | None = None) -> SearchResult:
        """Pick actions for ``player``.

        ``state`` is the real one, and it is used for exactly two things: the
        observation this player is entitled to, and the shape a determinization
        is built on. Nothing else here reads it.
        """
        draw = cursor if cursor is not None else Rng.from_seed(0).cursor()
        observation = Observation.of(state, player)

        options = joint_actions(state, player, self.config.max_branching,
                                self.config.switch_matchup,
                                self.config.switch_promise)
        if len(options) <= 1:
            only = options[0] if options else (Action.PASS,) * decisions_wanted(state, player)
            return SearchResult(only, ((only, 1.0),), 0.0, 0)

        if self.evaluator is not None:
            self.evaluator.reset()
        root = self._node(state, player)
        bounds = MinMax()
        per_draw = max(1, self.config.iterations // max(1, self.config.determinizations))
        # Batching needs a network to batch for. The heuristic is microseconds;
        # collecting it into batches would only add the virtual-loss diffusion
        # and buy nothing.
        batch = self.config.leaf_batch if self.evaluator is not None else 1
        done = 0
        while done < self.config.iterations:
            sampled = determinize(observation, state, draw,
                                  self.config.belief)
            left = min(per_draw, self.config.iterations - done)
            while left > 0:
                if batch > 1:
                    ran = self._simulate_batch(sampled, root, player, draw,
                                               bounds, min(batch, left))
                else:
                    self._simulate(sampled.clone(), root, player, draw, bounds)
                    ran = 1
                done += ran
                left -= ran

        counts = root.counts[0]
        best = max(range(len(counts)), key=lambda index: counts[index])
        total = sum(counts) or 1
        distribution = tuple(
            (root.options[0][index], counts[index] / total)
            for index in range(len(counts))
        )
        return SearchResult(root.options[0][best], distribution,
                            root.mean(0, best), done)

    # -- one simulation ----------------------------------------------------- #

    def _simulate(self, state: BattleState, node: Node, player: int,
                  cursor: RngCursor, bounds: MinMax | None = None) -> float:
        """One simulation: descend the tree, add **one** node, evaluate, back up.

        The one-node rule is not a detail. Recursing to the depth limit and
        creating a node at every level made each simulation build a whole new
        branch, so two hundred simulations created two thousand nodes that were
        each visited about once. Nothing was ever revisited, which is the entire
        mechanism of MCTS -- and the cost was paid twice over, since building a
        node means enumerating both sides' legal actions and asking for two
        priors.

        Descending through what already exists and expanding only the first
        unseen position turns that into two hundred nodes. It is both the
        textbook algorithm and, incidentally, the reason the network looked
        expensive: the forwards were five per cent of a decision that was
        mostly node construction.
        """
        path: list[tuple[Node, tuple[int, int]]] = []
        current = node
        depth = 0

        while True:
            if state.finished:
                value = terminal_value(state, player)
                break
            if depth >= self.config.max_depth or not current.expanded:
                value = self._evaluate(state, player, cursor)
                break

            mine = self._select(current, 0, bounds)
            theirs = (self._sample(current, 1, cursor) if self.config.sample_opponent
                      else self._select(current, 1, bounds))
            picked = (mine, theirs)
            path.append((current, picked))

            choices = (current.options[0][mine], current.options[1][theirs])
            # ``player`` is index 0 of every node's option lists by
            # construction, so the pair goes back the other way for the engine.
            ordered = choices if player == 0 else (choices[1], choices[0])
            try:
                state, _ = step(state, ordered[0], ordered[1])
            except IllegalActionError:
                # A determinization can disagree with the tree about what is
                # legal -- a resampled Pokemon knows different moves. Treat the
                # line as unplayable rather than letting it poison the counts.
                value = self._evaluate(state, player, cursor)
                break

            # The child key is what decides how wide the tree is. Our action
            # alone means the opponent's choice is folded into the node's value
            # as noise -- which is exactly what it is from where we sit, since
            # we do not get to see it before committing.
            key = (mine, 0) if self.config.sample_opponent else picked
            child = current.children.get(key)
            if child is None:
                # The one new node this simulation is allowed. Evaluate it and
                # stop: it has no statistics to descend on yet.
                child = current.children[key] = self._node(state, player)
                value = self._evaluate(state, player, cursor)
                break
            current = child
            depth += 1

        if bounds is not None:
            bounds.add(value)
        for visited, picked in path:
            visited.visits += 1
            for side in (0, 1):
                index = picked[side]
                visited.counts[side][index] += 1
                # Index 0 is always ``player``; the other side maximises the
                # negation, which is what makes the opponent in the tree play
                # well rather than randomly.
                visited.totals[side][index] += value if side == 0 else -value
        return value

    def _simulate_batch(self, sampled: BattleState, node: Node, player: int,
                        cursor: RngCursor, bounds: MinMax, count: int) -> int:
        """``count`` simulations whose leaves share one network forward.

        Three phases. **Descend**: each simulation selects a path exactly as
        ``_simulate`` does, but instead of stopping to evaluate it parks its
        leaf and stamps virtual loss down its path, steering the next
        simulation in the batch elsewhere. **Evaluate**: every parked leaf
        goes to the network in one call -- two rows per expansion (one per
        side), one row per depth-limited or illegal leaf. **Back up**: the
        virtual loss is refunded and the real value goes in, leaving exactly
        the statistics ``count`` sequential simulations would have left, up to
        which lines the in-flight pessimism steered them down.
        """
        pending = []  # (path, kind, payload, parent, key)
        for _ in range(count):
            state = sampled.clone()
            path: list[tuple[Node, tuple[int, int]]] = []
            current = node
            depth = 0
            while True:
                if state.finished:
                    pending.append((path, "terminal",
                                    terminal_value(state, player), None, None))
                    break
                if depth >= self.config.max_depth or not current.expanded:
                    pending.append((path, "value", state, None, None))
                    break
                mine = self._select(current, 0, bounds)
                theirs = (self._sample(current, 1, cursor)
                          if self.config.sample_opponent
                          else self._select(current, 1, bounds))
                picked = (mine, theirs)
                path.append((current, picked))
                choices = (current.options[0][mine], current.options[1][theirs])
                ordered = choices if player == 0 else (choices[1], choices[0])
                try:
                    state, _ = step(state, ordered[0], ordered[1])
                except IllegalActionError:
                    pending.append((path, "value", state, None, None))
                    break
                key = (mine, 0) if self.config.sample_opponent else picked
                child = current.children.get(key)
                if child is None:
                    pending.append((path, "expand", state, current, key))
                    break
                current = child
                depth += 1

            # Applied now, so the NEXT simulation in this batch sees this line
            # as taken-and-lost rather than untouched.
            for visited, chosen in path:
                visited.visits += 1
                for side in (0, 1):
                    visited.counts[side][chosen[side]] += 1
                    visited.totals[side][chosen[side]] -= VIRTUAL_LOSS

        # One forward for every leaf that needs the network.
        asks: list[tuple[BattleState, int]] = []
        for path, kind, payload, parent, key in pending:
            if kind == "value":
                asks.append((payload, player))
            elif kind == "expand":
                asks.append((payload, player))
                asks.append((payload, 1 - player))
        answers = iter(self.evaluator.look_many(asks))

        for path, kind, payload, parent, key in pending:
            if kind == "terminal":
                value = payload
            elif kind == "value":
                _, raw = next(answers)
                value = self.evaluator.value_from(raw, payload, player)
            else:
                probabilities, raw = next(answers)
                their_probabilities, _ = next(answers)
                value = self.evaluator.value_from(raw, payload, player)
                # Two simulations in one batch can race to the same child --
                # virtual loss discourages it but cannot forbid it. The first
                # builds the node; the second only backs its value up.
                if key not in parent.children:
                    fallback = [(Action.PASS,) * max(1, decisions_wanted(payload, player))]
                    mine_options = joint_actions(
                        payload, player, self.config.max_branching,
                        self.config.switch_matchup) or fallback
                    their_options = joint_actions(
                        payload, 1 - player, self.config.max_branching,
                        self.config.switch_matchup) or fallback
                    parent.children[key] = Node(
                        (mine_options, their_options),
                        priors=(self.evaluator.prior_from(
                                    probabilities, payload, player, mine_options),
                                self.evaluator.prior_from(
                                    their_probabilities, payload,
                                    1 - player, their_options)))

            bounds.add(value)
            # Counts and visits were already advanced with the virtual loss;
            # refund it and add what the leaf was actually worth.
            for visited, chosen in path:
                for side in (0, 1):
                    index = chosen[side]
                    visited.totals[side][index] += VIRTUAL_LOSS + (
                        value if side == 0 else -value)
        return len(pending)

    def _sample(self, node: Node, side: int, cursor: RngCursor) -> int:
        """Draw one of this side's actions from what the search believes it plays.

        Its own visit counts, which are a policy: an action the search has found
        good for that side has been visited more, so it comes up more. Falling
        back to the prior while there are no counts is what stops the first
        visits all going to option zero.
        """
        counts = node.counts[side]
        total = sum(counts)
        weights = ([count / total for count in counts] if total
                   else node.priors[side])
        roll = cursor.between(0, 999) / 1000.0
        cumulative = 0.0
        for index, weight in enumerate(weights):
            cumulative += weight
            if roll < cumulative:
                return index
        return len(weights) - 1

    def _select(self, node: Node, side: int,
                bounds: MinMax | None = None) -> int:
        """PUCT over this side's own marginals.

        Unvisited options are no longer taken in list order -- with a hundred
        joint actions in doubles that alone would eat the budget. The prior
        decides which are worth a first look, which is the same thing it does
        for every visit after.
        """
        counts = node.counts[side]
        totals = node.totals[side]
        priors = node.priors[side]
        parent = math.sqrt(max(1, node.visits))

        best_index, best_score = 0, -math.inf
        for index, count in enumerate(counts):
            mean = totals[index] / count if count else 0.0
            if bounds is not None and self.config.normalize_value:
                mean = bounds.scale(mean, side)
            # AlphaZero's exploration term: the prior decides where the early
            # visits go, and its pull decays as the statistics arrive.
            bias = self.config.prior_weight * priors[index] * parent / (1 + count)
            score = mean + bias
            if count:
                score += self.config.exploration * math.sqrt(
                    math.log(max(2, node.visits)) / count)
            else:
                # Nothing to average yet, so the prior is the whole answer --
                # plus enough to outrank anything already visited, so every
                # option gets one look before any gets a second.
                score += UNVISITED_BONUS
            if score > best_score:
                best_index, best_score = index, score
        return best_index

    # -- leaves ------------------------------------------------------------- #

    def _evaluate(self, state: BattleState, player: int, cursor: RngCursor) -> float:
        if self.evaluator is not None:
            return self.evaluator.value(state, player)
        if self.config.rollout_turns <= 0:
            return self._leaf(state, player)

        from pkcm.search.policy import play_out

        rollout = RandomPolicy(cursor)
        limit = state.turn + self.config.rollout_turns
        finished = play_out(state, (rollout, rollout), turn_limit=limit)
        if finished.finished:
            return terminal_value(finished, player)
        return self._leaf(finished, player)

    def _leaf(self, state: BattleState, player: int) -> float:
        """The handcrafted leaf value this search was configured with."""
        if self.config.evaluation == "pressure":
            return pressure(state, player, self.config.pressure_weight)
        if self.config.evaluation == "blind":
            # Terminals still pay, or the search would have no reason to
            # prefer winning. Everything short of one reads as unknown.
            return terminal_value(state, player) if state.finished else 0.0
        return heuristic(state, player)

    def _node(self, state: BattleState, player: int) -> Node:
        """Options with ``player`` first, so index 0 is always the searcher."""
        fallback = [(Action.PASS,) * max(1, decisions_wanted(state, player))]
        mine = joint_actions(state, player, self.config.max_branching,
                             self.config.switch_matchup,
                             self.config.switch_promise) or fallback
        theirs = joint_actions(state, 1 - player, self.config.max_branching,
                               self.config.switch_matchup,
                               self.config.switch_promise) or fallback
        return Node((mine, theirs),
                    priors=(self._prior(state, player, mine),
                            self._prior(state, 1 - player, theirs)))

    def _prior(self, state: BattleState, player: int, options: list) -> list[float]:
        if self.evaluator is not None:
            return self.evaluator.prior(state, player, options)
        return prior_over(state, player, options,
                          self.config.switch_matchup,
                          self.config.switch_promise)
