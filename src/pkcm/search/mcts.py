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
from pkcm.search.evaluate import heuristic, terminal_value
from pkcm.search.policy import (
    RandomPolicy,
    decisions_wanted,
    joint_actions,
    prior_over,
)

Choice = tuple[Action, ...]

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

    #: Total simulations. The only real dial.
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

        options = joint_actions(state, player, self.config.max_branching)
        if len(options) <= 1:
            only = options[0] if options else (Action.PASS,) * decisions_wanted(state, player)
            return SearchResult(only, ((only, 1.0),), 0.0, 0)

        if self.evaluator is not None:
            self.evaluator.reset()
        root = self._node(state, player)
        bounds = MinMax()
        per_draw = max(1, self.config.iterations // max(1, self.config.determinizations))
        done = 0
        while done < self.config.iterations:
            sampled = determinize(observation, state, draw)
            for _ in range(min(per_draw, self.config.iterations - done)):
                self._simulate(sampled.clone(), root, player, draw, bounds)
                done += 1

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
            return heuristic(state, player)

        from pkcm.search.policy import play_out

        rollout = RandomPolicy(cursor)
        limit = state.turn + self.config.rollout_turns
        finished = play_out(state, (rollout, rollout), turn_limit=limit)
        if finished.finished:
            return terminal_value(finished, player)
        return heuristic(finished, player)

    def _node(self, state: BattleState, player: int) -> Node:
        """Options with ``player`` first, so index 0 is always the searcher."""
        fallback = [(Action.PASS,) * max(1, decisions_wanted(state, player))]
        mine = joint_actions(state, player, self.config.max_branching) or fallback
        theirs = joint_actions(state, 1 - player, self.config.max_branching) or fallback
        return Node((mine, theirs),
                    priors=(self._prior(state, player, mine),
                            self._prior(state, 1 - player, theirs)))

    def _prior(self, state: BattleState, player: int, options: list) -> list[float]:
        if self.evaluator is not None:
            return self.evaluator.prior(state, player, options)
        return prior_over(state, player, options)
