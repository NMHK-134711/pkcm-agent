"""Every imported party against every other, one agent driving both sides.

``matchup`` asks which *agent* is stronger and holds the teams constant by
drawing them from one distribution. This asks the opposite question -- which
*team* is stronger -- and so holds the agent constant instead: the same search,
the same iteration count, the same seeds on both seats. The only thing that
differs across a fixture is the six Pokemon.

Two things that would otherwise decide the answer are taken away:

* **Seat.** Every fixture is played both ways round with the same battle seed,
  so moving first is worth the same to both entrants.
* **The rolls.** Repeat *r* uses the same base seed in every fixture, which is
  common random numbers: when a team wins on repeat 3 it is not because repeat
  3 was kind to it and cruel to no one else.

What this cannot take away is the field. A round robin between twenty teams
measures a team against those nineteen, and the nineteen are the archive we
happen to have rather than the ladder. A team that beats this field is a team
that beats this field.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterator, Sequence

from pkcm.data.dex import Dex, load_dex
from pkcm.engine.legality import Party, ranker_parties
from pkcm.engine.rng import Rng
from pkcm.engine.state import BattleConfig, new_battle
from pkcm.search import MCTS, SearchConfig
from pkcm.search.policy import SearchPolicy, play_out
from pkcm.train.parallel import default_workers, map_unordered


@dataclass(frozen=True, slots=True)
class TournamentConfig:
    """One agent, and the field it plays the whole tournament with."""

    battle_format: str = "singles"
    regulation: str = "m_b"
    search: SearchConfig = field(default_factory=SearchConfig)
    #: A saved network for the prior and leaf value, on **both** sides. There is
    #: no side A here -- an asymmetric agent would price the seat, not the team.
    checkpoint: str | None = None
    trust: float = 1.0
    #: Where the parties come from. ``None`` is the committed archive.
    parties: str | None = None
    #: Base for the battle seeds.
    seed: int = 500_000


@dataclass(frozen=True, slots=True)
class Result:
    """What happened when ``a`` and ``b`` met, over both seatings."""

    a: int
    b: int
    a_wins: int = 0
    b_wins: int = 0
    draws: int = 0


@dataclass(frozen=True, slots=True)
class Standing:
    """One entrant's total across the field."""

    party: int
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def decided(self) -> int:
        return self.wins + self.losses

    @property
    def rate(self) -> float:
        return self.wins / self.decided if self.decided else 0.0


def fixtures(entrants: Sequence[int], repeats: int) -> list[tuple[int, int, int]]:
    """Every unordered pair, ``repeats`` times each.

    Mirrors are left out: a team against itself is 50% by construction and
    would only dilute the interval.
    """
    return [(a, b, r)
            for i, a in enumerate(entrants)
            for b in entrants[i + 1:]
            for r in range(repeats)]


def play_fixture(dex: Dex, parties: Sequence[Party], config: TournamentConfig,
                 fixture: tuple[int, int, int]) -> Result:
    """One pair of parties, played both ways round on the same seed."""
    a, b, repeat = fixture
    battle_config = BattleConfig(dex=dex,
                                 regulation=dex.regulation(config.regulation),
                                 battle_format=config.battle_format)
    evaluator = _evaluator(dex, config)
    seed = config.seed + repeat

    result = Result(a=a, b=b)
    for swap in (False, True):
        # The seats keep their own policy cursors and the *teams* move between
        # them, so neither entrant is always the one holding cursor ``seed``.
        first = SearchPolicy(MCTS(config.search, evaluator=evaluator),
                             Rng.from_seed(seed).cursor())
        second = SearchPolicy(MCTS(config.search, evaluator=evaluator),
                              Rng.from_seed(seed + 7777).cursor())
        teams = ((parties[b].team, parties[a].team) if swap
                 else (parties[a].team, parties[b].team))
        state = play_out(new_battle(battle_config, teams, seed=seed),
                         (first, second))

        a_seat = 1 if swap else 0
        if state.winner is None:
            result = Result(a, b, result.a_wins, result.b_wins, result.draws + 1)
        elif state.winner == a_seat:
            result = Result(a, b, result.a_wins + 1, result.b_wins, result.draws)
        else:
            result = Result(a, b, result.a_wins, result.b_wins + 1, result.draws)
    return result


def stream(config: TournamentConfig, schedule: Sequence[tuple[int, int, int]],
           workers: int | None = None) -> Iterator[Result]:
    """Each fixture's result as it lands, so a caller can report progress."""
    count = workers if workers is not None else default_workers()
    if count <= 1:
        # Same reason as ``matchup.stream``: a CPU matmul reduces in a different
        # order on twenty threads than on one, and an argmax that flips there
        # sends the battle somewhere else. The serial path has to agree with the
        # pooled one or it cannot check it.
        try:
            import torch

            torch.set_num_threads(1)
        except ImportError:  # pragma: no cover - no network to evaluate
            pass
        dex = load_dex()
        parties = ranker_parties(config.parties)
        for fixture in schedule:
            yield play_fixture(dex, parties, config, fixture)
        return

    yield from map_unordered(_play, schedule, initializer=_start_worker,
                             initargs=(config,), workers=count, what="fixture")


def standings(results: Sequence[Result],
              entrants: Sequence[int]) -> list[Standing]:
    """Fold the fixtures into one row per entrant, best first."""
    wins = {party: 0 for party in entrants}
    losses = dict.fromkeys(wins, 0)
    draws = dict.fromkeys(wins, 0)
    for one in results:
        wins[one.a] += one.a_wins
        wins[one.b] += one.b_wins
        losses[one.a] += one.b_wins
        losses[one.b] += one.a_wins
        draws[one.a] += one.draws
        draws[one.b] += one.draws
    rows = [Standing(party, wins[party], losses[party], draws[party])
            for party in entrants]
    return sorted(rows, key=lambda row: (-row.rate, -row.decided, row.party))


# -- worker state ---------------------------------------------------------- #

_DEX: Dex | None = None
_PARTIES: tuple[Party, ...] | None = None
_CONFIG: TournamentConfig | None = None
_CACHED: tuple[str, float, int, object] | None = None


def _start_worker(config: TournamentConfig) -> None:
    global _DEX, _PARTIES, _CONFIG
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:  # pragma: no cover - torch is required for a checkpoint
        pass
    _DEX = load_dex()
    _PARTIES = ranker_parties(config.parties)
    _CONFIG = config


def _play(fixture: tuple[int, int, int]) -> Result:
    assert _DEX is not None and _PARTIES is not None and _CONFIG is not None, \
        "worker was not initialised"
    return play_fixture(_DEX, _PARTIES, _CONFIG, fixture)


def _evaluator(dex: Dex, config: TournamentConfig):
    """The network, loaded once per process rather than once per battle."""
    if config.checkpoint is None:
        return None
    global _CACHED
    # Keyed by the dex as well: ``sheet_for`` and the vocabulary are built
    # against one dex object, and an evaluator cached under a different one
    # answers with the wrong tables.
    key = (config.checkpoint, config.trust, id(dex))
    if _CACHED is not None and _CACHED[:3] == key:
        return _CACHED[3]

    from pkcm.envs.encoding import SCALAR_SIZE, action_space_size
    from pkcm.train.evaluator import from_checkpoint

    battle_config = BattleConfig(dex=dex,
                                 regulation=dex.regulation(config.regulation),
                                 battle_format=config.battle_format)
    built = from_checkpoint(
        config.checkpoint, dex,
        action_space_size(battle_config.registered, battle_config.brought),
        SCALAR_SIZE, device="cpu", trust=config.trust)
    _CACHED = (config.checkpoint, config.trust, id(dex), built)
    return built
