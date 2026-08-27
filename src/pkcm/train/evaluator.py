"""The network, wearing the two shapes the search already expects.

``MCTS`` asks for exactly two things it cannot work out itself:

    a prior over this node's options   -- ``search.policy.prior_over``
    a value for this leaf              -- ``search.evaluate.heuristic``

An ``Evaluator`` answers both. Handing one to ``MCTS`` swaps the heuristics for
the network without the tree knowing anything changed, which is why those two
functions were kept as functions.

The leak rule holds here too, and it matters more than anywhere else: the
network is shown ``Observation.of(state, player)`` and never the state. Inside
the search that state is a determinization, so it has already committed to a
guess about everything hidden -- but a network trained on the truth would learn
to read the opponent's team, and would then be worthless against a real one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from pkcm.data.dex import Dex
from pkcm.engine.state import BattleState
from pkcm.envs.encoding import (
    Vocabulary,
    encode_action,
    encode_observation,
)
from pkcm.envs.observation import Observation
from pkcm.envs.reference import sheet_for
from pkcm.search.evaluate import heuristic
from pkcm.train.net import ChampionsNet


@dataclass
class Evaluator:
    """Network-backed prior and value, cached within one decision."""

    net: ChampionsNet
    dex: Dex
    device: torch.device | str = "cpu"
    #: Blend with the handcrafted prior **and the handcrafted leaf value**.
    #: One is all network; zero is all heuristic.
    #:
    #: The docstring here used to promise exactly that and only deliver half of
    #: it: ``prior`` blended, ``value`` returned the network's number whatever
    #: ``trust`` said. So a network the loop had explicitly decided to half
    #: believe still had complete authority over every leaf in the tree -- and
    #: the leaf value is the half that was broken. Measured at iteration 1, the
    #: value head was emitting +-0.99 on team preview positions, where the
    #: honest answer is "nobody has moved yet", and the search it drove lost to
    #: the handcrafted one 16.2% [9.8, 25.8].
    trust: float = 1.0
    _vocabulary: Vocabulary | None = field(default=None, repr=False)
    _sheet: object | None = field(default=None, repr=False)
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self._vocabulary is None:
            self._vocabulary = Vocabulary.of(self.dex)
        if self._sheet is None:
            self._sheet = sheet_for(self.dex, self._vocabulary)
        self.net.eval()

    # -- what the tree asks for -------------------------------------------- #

    def prior(self, state: BattleState, player: int, options: list) -> list[float]:
        """A distribution over ``options``, from the policy head."""
        probabilities, _ = self._look(state, player)
        registered = state.config.registered
        brought = state.config.brought

        scores = []
        for choice in options:
            # A doubles choice is several actions; the head scores each one, so
            # the choice is worth their mean. Crude, and it is the same
            # compromise the training target makes when it splits a joint
            # visit count across positions.
            indices = [encode_action(action, registered, brought) for action in choice]
            scores.append(float(np.mean([probabilities[i] for i in indices])) if indices
                          else 0.0)

        total = sum(scores)
        if total <= 0:
            return [1.0 / max(1, len(options))] * len(options)
        network = [score / total for score in scores]
        if self.trust >= 1.0:
            return network

        from pkcm.search.policy import prior_over

        handcrafted = prior_over(state, player, options)
        return [self.trust * a + (1 - self.trust) * b
                for a, b in zip(network, handcrafted)]

    def value(self, state: BattleState, player: int) -> float:
        """How good this position looks, in ``[-1, 1]``.

        Blended with ``evaluate.heuristic`` on the same terms as the prior. The
        heuristic is blunt and it is compressed, but it is never confidently
        wrong, and a saturated value head is worse than a blunt one: it does not
        merely fail to rank the lines, it ranks them backwards with conviction.
        """
        _, value = self._look(state, player)
        if self.trust >= 1.0:
            return float(value)
        return (self.trust * float(value)
                + (1 - self.trust) * heuristic(state, player))

    # -- one forward pass, reused ------------------------------------------ #

    def _look(self, state: BattleState, player: int) -> tuple[np.ndarray, float]:
        # Keyed by ``id(state)`` **and holding the state alive**. A search
        # creates and drops determinizations by the hundred inside one decision,
        # CPython reuses an address the moment one is collected, and the next
        # state to land there would be answered with its predecessor's
        # evaluation. Nothing fails: the tree simply gets a wrong number
        # sometimes, and *which* times depends on allocation order, so the same
        # seed gave different battles in a worker than in the parent.
        key = (id(state), player, state.turn, state.phase)
        found = self._cache.get(key)
        if found is not None and found[0] is state:
            return found[1]

        observation = Observation.of(state, player)
        encoded = encode_observation(observation, self._vocabulary, self._sheet, self.dex)
        probabilities, values = self.net.evaluate([encoded], self.device)
        found = (probabilities[0], float(values[0]))
        # Bounded: a decision touches a few hundred positions and the cache is
        # only useful within one. Left to grow it would outlive its usefulness
        # and hold every state a self-play game ever visited.
        if len(self._cache) > 4096:
            self._cache.clear()
        self._cache[key] = (state, found)
        return found

    def reset(self) -> None:
        self._cache.clear()


def from_checkpoint(path, dex: Dex, action_space: int, scalar_size: int,
                    device: torch.device | str = "cpu", trust: float = 1.0) -> Evaluator:
    """Rebuild an evaluator from a saved network."""
    from pkcm.train.net import build
    from pkcm.train.trainer import load_into

    vocabulary = Vocabulary.of(dex)
    sheet = sheet_for(dex, vocabulary)
    payload = torch.load(path, map_location=device, weights_only=False)
    net = build(vocabulary, sheet, payload["action_space"], scalar_size,
                payload.get("config"))
    load_into(net, path, torch.device(device))
    return Evaluator(net=net, dex=dex, device=device, trust=trust,
                     _vocabulary=vocabulary, _sheet=sheet)
