"""The loop's own arena, across every core.

``train_loop`` has to answer one question per evaluation: does the search driven
by the network beat the search driven by the handcrafted prior? That is the only
number in the loop that knows anything -- the losses fall whatever happens --
and it was the slowest thing in the run by a wide margin, because it played its
games one at a time while nineteen cores watched.

Self-play was parallelised for exactly this reason and this is the same shape:
battles do not talk to each other. The seeds are fixed per match, so a battle
plays the same game whichever worker picks it up and whether or not there is a
pool at all. ``workers=1`` runs in this process and returns the same counts,
which is what makes the parallel version checkable rather than merely faster.

Both seatings of every pair of teams, as everywhere else in this project: a win
rate that comes from having drawn the better team is not a win rate.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterator

from pkcm.data.dex import Dex, load_dex
from pkcm.engine.legality import make_team
from pkcm.engine.rng import Rng
from pkcm.engine.state import BattleConfig, new_battle
from pkcm.search import MCTS, SearchConfig
from pkcm.search.policy import SearchPolicy, play_out


@dataclass(frozen=True, slots=True)
class MatchConfig:
    """Which two searches, over which format.

    ``checkpoint`` is the only difference between the sides: one search gets the
    network as its prior and leaf value, the other gets the handcrafted pair.
    Everything else -- iterations, determinizations, teams, seats -- is shared,
    so the comparison is of the evaluation and nothing else.
    """

    #: A saved network for side A's prior and leaf value. ``None`` puts the
    #: handcrafted pair on both sides, which is how one search configuration is
    #: measured against another.
    checkpoint: str | None = None
    battle_format: str = "singles"
    regulation: str = "m_b"
    #: Which distribution the teams come from -- ``random`` or
    #: ``ranker``. Random teams are the null distribution and a long
    #: way from the game: 37.5% of their Pokemon carry no same-type
    #: attack at all, against 4.9% of the ranker pool's.
    teams: str = "random"
    #: The opponent seat's distribution. ``None`` follows ``teams``. Both
    #: seatings are played with the pair held fixed, so an asymmetric draw
    #: still has each side pilot each team once.
    foe_teams: str | None = None
    search: SearchConfig = field(default_factory=SearchConfig)
    trust: float = 1.0
    #: Per-head overrides. ``None`` follows ``trust``. Setting one to 1 and the
    #: other to 0 asks which head is responsible for a change in strength.
    trust_prior: float | None = None
    trust_value: float | None = None
    #: Which pool the belief draws from -- ``"ranker"`` or ``"invented"``.
    #: Applies to **both** sides, because it is a property of the world both
    #: are playing in rather than of either player. See ``envs.belief``.
    belief_pool: str = "ranker"
    #: Side B's search, when the two differ. ``None`` gives it side A's.
    search_b: SearchConfig | None = None
    #: Side B's network. ``False`` gives side B the handcrafted pair,
    #: ``True`` the same network as side A, and a **path** its own network --
    #: which is what gating needs: candidate against incumbent, nothing shared.
    checkpoint_b: bool | str = False
    #: Base for the team seeds. Held away from the self-play seeds so the
    #: measurement is not played on teams the network was trained on.
    team_seed: int = 90000


@dataclass(frozen=True, slots=True)
class Record:
    """Who won, over however many games were decided."""

    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def decided(self) -> int:
        return self.wins + self.losses

    def __add__(self, other: "Record") -> "Record":
        return Record(self.wins + other.wins, self.losses + other.losses,
                      self.draws + other.draws)


def play_match(dex: Dex, config: MatchConfig, match: int) -> Record:
    """One pair of teams, played both ways round.

    Returns the network side's record. The seeds are derived from ``match``
    alone, which is what lets the pool hand matches out in any order.
    """
    from pkcm.envs import belief as _belief

    # Process-global and cached, so a worker has to be told once.
    if _belief._SOURCE != config.belief_pool:
        _belief.use_pool(config.belief_pool)

    battle_config = BattleConfig(dex=dex, regulation=dex.regulation(config.regulation),
                                 battle_format=config.battle_format)
    teams = tuple(
        make_team(dex, battle_config.regulation,
                  Rng.from_seed(config.team_seed + match * 2 + offset).cursor(),
                  config.battle_format,
                  config.teams if offset == 1 else
                  (config.foe_teams or config.teams))
        for offset in (1, 2)
    )
    evaluator = _evaluator(dex, config, config.checkpoint) if config.checkpoint else None
    other = config.search_b or config.search
    if isinstance(config.checkpoint_b, str):
        evaluator_b = _evaluator(dex, config, config.checkpoint_b)
    else:
        evaluator_b = evaluator if config.checkpoint_b else None

    record = Record()
    for swap in (False, True):
        netted = SearchPolicy(MCTS(config.search, evaluator=evaluator),
                              Rng.from_seed(match).cursor())
        plain = SearchPolicy(MCTS(other, evaluator=evaluator_b),
                             Rng.from_seed(match + 7777).cursor())
        policies = (plain, netted) if swap else (netted, plain)
        state = play_out(new_battle(battle_config, teams, seed=match), policies)

        net_side = 1 if swap else 0
        if state.winner is None:
            record += Record(draws=1)
        elif state.winner == net_side:
            record += Record(wins=1)
        else:
            record += Record(losses=1)
    return record


def measure(config: MatchConfig, matches: int, workers: int | None = None) -> Record:
    """Play ``matches`` pairs of teams -- two battles each -- and total them up."""
    return sum(stream(config, matches, workers), Record())


def stream(config: MatchConfig, matches: int,
           workers: int | None = None) -> Iterator[Record]:
    """Each match's record as it lands, so a caller can report progress."""
    count = workers if workers is not None else default_workers()
    if count <= 1:
        # Match the workers' thread count, which is not a performance choice
        # here: a CPU matmul reduces in a different order on twenty threads than
        # on one, the last bits of the network's output differ, and every so
        # often an argmax flips and the battle diverges. Measured, this path
        # disagreed with the pooled one about one run in three -- and the run
        # that is "right" is neither, so the two have to be made the same.
        try:
            import torch

            torch.set_num_threads(1)
        except ImportError:  # pragma: no cover - no network to evaluate
            pass
        dex = load_dex()
        for match in range(matches):
            yield play_match(dex, config, match)
        return

    from pkcm.train.parallel import map_unordered

    # A dropped match is a smaller sample, not a wrong one: the seeds are fixed
    # per match, so what does come back is exactly what a serial run would have
    # produced, and ``Record.decided`` carries the honest denominator into the
    # confidence interval.
    yield from map_unordered(_play, range(matches), initializer=_start_worker,
                             initargs=(config,), workers=count, what="match")


def default_workers() -> int:
    from pkcm.train.parallel import default_workers as shared

    return shared()


# -- worker state ---------------------------------------------------------- #

#: Built once per worker. Windows spawns, so a worker starts from nothing and
#: would otherwise reload the dex and the checkpoint for every battle.
_DEX: Dex | None = None
_CONFIG: MatchConfig | None = None
_CACHED: dict | None = None


def _start_worker(config: MatchConfig) -> None:
    global _DEX, _CONFIG
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        import torch

        # One process per core already saturates the machine. A network this
        # small gains nothing from intra-op threads and the oversubscription
        # costs more than the forwards do.
        torch.set_num_threads(1)
    except ImportError:  # pragma: no cover - torch is required to have a checkpoint
        pass
    _DEX = load_dex()
    _CONFIG = config


def _play(match: int) -> Record:
    assert _DEX is not None and _CONFIG is not None, "worker was not initialised"
    return play_match(_DEX, _CONFIG, match)


def _evaluator(dex: Dex, config: MatchConfig, checkpoint: str):
    """A network, loaded once per process per path rather than once per battle.

    A dict rather than a single slot, because gating puts two different
    networks in one match -- candidate on side A, incumbent on side B -- and a
    one-slot cache would reload them alternately every battle. Keyed by the
    dex too: ``sheet_for`` and the vocabulary are built against one dex
    object, so an evaluator cached under a different one answers with the
    wrong tables.
    """
    global _CACHED
    if not isinstance(_CACHED, dict):
        _CACHED = {}
    key = (checkpoint, config.trust, config.trust_prior,
           config.trust_value, id(dex))
    found = _CACHED.get(key)
    if found is not None:
        return found

    from pkcm.envs.encoding import SCALAR_SIZE, action_space_size
    from pkcm.train.evaluator import from_checkpoint

    battle_config = BattleConfig(dex=dex, regulation=dex.regulation(config.regulation),
                                 battle_format=config.battle_format)
    built = from_checkpoint(
        checkpoint, dex,
        action_space_size(battle_config.registered, battle_config.brought),
        SCALAR_SIZE, device="cpu", trust=config.trust,
        trust_prior=config.trust_prior, trust_value=config.trust_value)
    _CACHED[key] = built
    return built
