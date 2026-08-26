"""One self-play battle, recorded as training data.

A sample is what AlphaZero trains on, and the shape is fixed by what the two
network heads need:

``observation``   what that player could see. Encoded arrays, so a worker hands
                  back something small and picklable rather than a battle state.
``policy``        the search's visit distribution over the flat action space.
                  The policy head learns to predict it, which is what lets the
                  network stand in for the search's prior next time round.
``value``         who eventually won, from that player's side. The value head
                  learns to predict it, which is what lets a leaf be judged
                  without playing thirty more turns.

Both players' decisions are recorded from every battle. They see different
information and reach different conclusions, so a battle is two games' worth of
data, not one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pkcm.data.dex import Dex
from pkcm.engine.battle import step
from pkcm.engine.legality import random_team
from pkcm.engine.rng import Rng, RngCursor
from pkcm.engine.state import BattleConfig, Phase, new_battle
from pkcm.envs.encoding import (
    Vocabulary,
    action_space_size,
    encode_action,
    encode_observation,
)
from pkcm.envs.observation import Observation
from pkcm.envs.reference import sheet_for
from pkcm.search import MCTS, SearchConfig


@dataclass(frozen=True, slots=True)
class Sample:
    """One decision, with what came of it."""

    observation: dict[str, np.ndarray]
    policy: np.ndarray
    value: float
    #: Which player, and which turn. Not fed to the network -- kept so a bad
    #: batch can be traced back to the battle that produced it.
    player: int
    turn: int


@dataclass(frozen=True, slots=True)
class SelfPlayConfig:
    battle_format: str = "singles"
    regulation: str = "m_b"
    search: SearchConfig = field(default_factory=SearchConfig)
    #: Stop a battle that will not end. The engine's own limit is 200 turns and
    #: reaching it is decided on attrition, which is a real result -- this is
    #: only a guard against a search that has gone wrong.
    max_turns: int = 220
    #: Record decisions only while there is something to decide. A forced switch
    #: with one legal option teaches the policy head nothing except that the
    #: mask exists.
    skip_forced: bool = True
    #: A saved network for the search to use. ``None`` runs the handcrafted
    #: prior and value, which is how the first iteration gets its data -- there
    #: is nothing to load yet, and the heuristic search already beats a one-turn
    #: calculator two games in three.
    checkpoint: str | None = None
    #: How far to trust the network over the heuristic prior. A freshly
    #: initialised network is worse than the heuristic, so early iterations
    #: blend rather than hand over.
    trust: float = 1.0


def play_one(dex: Dex, config: SelfPlayConfig, seed: int) -> list[Sample]:
    """Play one battle, search against search, and return what both sides saw."""
    battle_config = BattleConfig(dex=dex, regulation=dex.regulation(config.regulation),
                                 battle_format=config.battle_format)
    teams = tuple(
        random_team(dex, battle_config.regulation,
                    Rng.from_seed(seed * 2 + offset).cursor(), config.battle_format)
        for offset in (1, 2)
    )
    state = new_battle(battle_config, teams, seed=seed)

    vocabulary = Vocabulary.of(dex)
    sheet = sheet_for(dex, vocabulary)
    search = MCTS(config.search, evaluator=_evaluator(dex, config))
    cursor = Rng.from_seed(seed ^ 0x5EED).cursor()
    width = action_space_size(battle_config.registered, battle_config.brought)

    pending: list[tuple[dict, np.ndarray, int, int]] = []
    while not state.finished and state.turn <= config.max_turns:
        chosen = []
        for player in (0, 1):
            result = search.choose(state, player, cursor)
            chosen.append(result.action)
            if config.skip_forced and result.iterations == 0:
                continue
            observation = Observation.of(state, player)
            pending.append((
                encode_observation(observation, vocabulary, sheet, dex),
                _policy_target(result, width, battle_config),
                player,
                state.turn,
            ))
        state, _ = step(state, chosen[0], chosen[1])

    return [
        Sample(observation=observation, policy=policy,
               value=_outcome(state, player), player=player, turn=turn)
        for observation, policy, player, turn in pending
    ]


#: One evaluator per worker process, not one per battle. Loading a checkpoint
#: and rebuilding the reference sheet costs more than a couple of battles do.
_EVALUATOR: tuple[str | None, float, object] | None = None


def _evaluator(dex: Dex, config: SelfPlayConfig):
    global _EVALUATOR
    if config.checkpoint is None:
        return None
    key = (config.checkpoint, config.trust)
    if _EVALUATOR is not None and _EVALUATOR[:2] == key:
        return _EVALUATOR[2]

    from pkcm.envs.encoding import SCALAR_SIZE
    from pkcm.train.evaluator import from_checkpoint

    battle_config = BattleConfig(dex=dex, regulation=dex.regulation(config.regulation),
                                 battle_format=config.battle_format)
    built = from_checkpoint(
        config.checkpoint, dex,
        action_space_size(battle_config.registered, battle_config.brought),
        SCALAR_SIZE, device="cpu", trust=config.trust)
    _EVALUATOR = (config.checkpoint, config.trust, built)
    return built


def _policy_target(result, width: int, battle_config: BattleConfig) -> np.ndarray:
    """The visit distribution, spread over the flat action space.

    A doubles choice is several actions at once, so its visit share is split
    evenly across the positions it covers. That is lossy -- it cannot express
    "this pair together" -- and it is what a per-position policy head can
    consume. A head that predicts joint actions would not need the split, and
    would have a few hundred outputs to predict instead.
    """
    target = np.zeros(width, dtype=np.float32)
    for choice, share in result.distribution:
        if not choice:
            continue
        each = share / len(choice)
        for action in choice:
            target[encode_action(action, battle_config.registered,
                                 battle_config.brought)] += each
    total = target.sum()
    return target / total if total > 0 else target


def _outcome(state, player: int) -> float:
    """What the environment would have paid. Unfinished battles score nothing."""
    if state.phase is not Phase.FINISHED or state.winner is None:
        return 0.0
    return 1.0 if state.winner == player else -1.0
