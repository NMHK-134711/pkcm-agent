"""Teach the network the handcrafted prior before asking it to beat one.

Run B settled the question the loop had been failing on. The search's visit
distribution reproduces its own prior, so the network learns a lossy copy of
``policy.prior_over`` -- and swapping that copy in makes the search *worse*.
Both runs started around 16-22% against the handcrafted search and neither
climbed out; the deficit is not something self-play iterations pay off,
because it is there on the first iteration and the loop's data comes from a
search that is already crippled by it.

So hand the network the prior directly, first, as supervised learning:

    observation  ->  ``prior_over`` over the same options the search enumerates
    observation  ->  ``evaluate.heuristic`` for that position

Then AlphaZero starts from a network that is worth what the handcrafted pair
is worth, swapping costs nothing, and whatever self-play adds is added rather
than spent climbing back to par. **The acceptance test is an arena at 50%, not
a low loss** -- a network that imitates the prior perfectly should be
indistinguishable from it, and anything below that means the copy is lossy in
a way the loop will then have to pay for.

The value head is pre-trained too, on the same terms. ``trust`` blends both
the prior and the leaf value, so leaving the value head untrained would hand
the tree a random number for every leaf the moment trust rose.

**States come from cheap play, not from search.** No search means thousands of
battles a minute rather than tens, and what is being learned here is a function
of the position -- the prior does not care how the position was reached. The
epsilon of random moves is there so the distribution is not a thin greedy
corridor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from pkcm.data.dex import Dex
from pkcm.engine.actions import Action
from pkcm.engine.battle import step
from pkcm.engine.legality import make_team
from pkcm.engine.rng import Rng
from pkcm.engine.state import BattleConfig, new_battle
from pkcm.envs.encoding import (
    Vocabulary,
    action_space_size,
    encode_observation,
)
from pkcm.envs.observation import Observation
from pkcm.envs.reference import sheet_for
from pkcm.search.evaluate import heuristic
from pkcm.search.policy import (
    GreedyPolicy,
    decisions_wanted,
    joint_actions,
    prior_over,
)
from pkcm.train.samples import Sample, policy_target


@dataclass(frozen=True, slots=True)
class ImitateConfig:
    battle_format: str = "singles"
    regulation: str = "m_b"
    #: Which distribution the teams come from -- ``random`` or
    #: ``ranker``. Random teams are the null distribution and a long
    #: way from the game: 37.5% of their Pokemon carry no same-type
    #: attack at all, against 4.9% of the ranker pool's.
    teams: str = "random"
    #: Must match the search's, because the prior is a distribution *over the
    #: options the search enumerates*. Trained on the full list and read back
    #: from a truncated one, the target would be a different function.
    max_branching: int = 24
    max_turns: int = 220
    #: How often to move at random instead of greedily. Greedy play alone walks
    #: a narrow corridor of positions and the network would be excellent there
    #: and untested everywhere the search actually goes.
    epsilon: float = 0.25
    #: Skip decisions with one legal option, as self-play does: there is nothing
    #: to imitate, and the mask already says so.
    skip_forced: bool = True
    #: Record the pick and stop, without playing the battle out.
    #:
    #: A battle hands over about twenty-six battle turns and **exactly two
    #: picks**, so a corpus balanced by battles is starved at the one decision
    #: the game is largest at. Measured on the first pre-trained network, the
    #: policy agreed with the handcrafted prior's top pick 2.5% of the time --
    #: worse than the 4.2% of guessing among twenty-four. Picks cost almost
    #: nothing to draw on their own, because the battle after them is the
    #: expensive part and it is not needed.
    preview_only: bool = False


def play_one(dex: Dex, config: ImitateConfig, seed: int) -> list[Sample]:
    """One cheap battle, recorded as (position -> handcrafted prior, value)."""
    battle_config = BattleConfig(dex=dex, regulation=dex.regulation(config.regulation),
                                 battle_format=config.battle_format)
    teams = tuple(
        make_team(dex, battle_config.regulation,
                  Rng.from_seed(seed * 2 + offset).cursor(), config.battle_format,
                  config.teams)
        for offset in (1, 2)
    )
    state = new_battle(battle_config, teams, seed=seed)

    vocabulary = Vocabulary.of(dex)
    sheet = sheet_for(dex, vocabulary)
    width = action_space_size(battle_config.registered, battle_config.brought)
    greedy = (GreedyPolicy.seeded(seed * 3 + 1), GreedyPolicy.seeded(seed * 3 + 2))
    cursor = Rng.from_seed(seed ^ 0x111111).cursor()

    collected: list[Sample] = []
    while not state.finished and state.turn <= config.max_turns:
        chosen = []
        for player in (0, 1):
            options = joint_actions(state, player, config.max_branching)
            if not options:
                chosen.append((Action.PASS,) * decisions_wanted(state, player))
                continue

            if not (config.skip_forced and len(options) == 1):
                prior = prior_over(state, player, options)
                collected.append(Sample(
                    observation=encode_observation(
                        Observation.of(state, player), vocabulary, sheet, dex),
                    policy=policy_target(zip(options, prior), width, battle_config),
                    # The handcrafted leaf value, not the eventual outcome. This
                    # phase is imitation: the target is what the search would
                    # have believed here, and it is a much steadier signal than
                    # who happened to win eleven turns later.
                    value=float(heuristic(state, player)),
                    player=player, turn=state.turn, battle=seed))

            if cursor.below(1000) < int(config.epsilon * 1000):
                chosen.append(options[cursor.between(0, len(options) - 1)])
            else:
                chosen.append(greedy[player].act(state, player))
        if config.preview_only:
            return collected
        state, _ = step(state, chosen[0], chosen[1])

    return collected


def baseline_mae(samples: list[Sample]) -> float:
    """What predicting a constant zero scores on these value targets.

    Self-play targets are +-1, so that baseline is exactly 1.0 and the printed
    error reads against it without thinking. The heuristic is material, which is
    usually near zero, so **the same number means something completely
    different here** and has to be quoted next to its own baseline.
    """
    if not samples:
        return 0.0
    return float(np.mean([abs(sample.value) for sample in samples]))


def baseline_policy_loss(samples: list[Sample]) -> float:
    """The cross-entropy floor: the mean entropy of the targets themselves.

    A cross-entropy of 1.58 says nothing on its own. If the targets average
    1.55 nats of entropy then 1.58 is nearly perfect imitation, and if they
    average 0.9 then the network has learned almost none of it. This project
    has been fooled by an unanchored loss before -- the first loop run sat at
    policy 1.675 for eight iterations, which turned out to be exactly its own
    targets' entropy, meaning there had been nothing left to learn all along.
    """
    if not samples:
        return 0.0
    total = []
    for sample in samples:
        share = sample.policy[sample.policy > 0]
        total.append(float(-(share * np.log(share)).sum()))
    return float(np.mean(total))


# -- across every core ----------------------------------------------------- #

_DEX: Dex | None = None
_CONFIG: ImitateConfig | None = None


def _start_worker(config: ImitateConfig) -> None:
    global _DEX, _CONFIG
    from pkcm.data.dex import load_dex

    _DEX = load_dex()
    _CONFIG = config


def _play(seed: int) -> list[Sample]:
    assert _DEX is not None and _CONFIG is not None, "worker was not initialised"
    return play_one(_DEX, _CONFIG, seed)


def generate(config: ImitateConfig, battles: int, seed: int = 0,
             workers: int | None = None) -> Iterator[list[Sample]]:
    """Play ``battles`` cheap games, yielding each one's samples as it lands."""
    from pkcm.data.dex import load_dex
    from pkcm.train.parallel import default_workers, map_unordered

    count = workers if workers is not None else default_workers()
    seeds = [seed + index for index in range(battles)]
    if count <= 1:
        dex = load_dex()
        for one in seeds:
            yield play_one(dex, config, one)
        return

    yield from map_unordered(_play, seeds, initializer=_start_worker,
                             initargs=(config,), workers=count, what="battle")
