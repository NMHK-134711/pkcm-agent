"""How a party does against its worst matchups, measured without paying for
the easy ones.

The field does not sort (``runs/tournament_contenders.json``): 43 beats 14
beats 42 beats 43, every edge separable. In a field like that a mean win rate
mostly reports which opponents happened to be in it, so it is the wrong thing
to optimise a party for. What survives the cycles is the floor -- whether a
party has an answer to most of what it will meet -- and with three of six
brought, "an answer" can be a different three rather than a different party.

**The floor is not the minimum.** Over 45 opponents the sample minimum is the
noisiest statistic available: with eight games a pair its standard error is
larger than the differences being ranked, so optimising it chases whichever
matchup was unluckiest rather than whichever is genuinely bad. This measures
the CVaR instead -- the mean of the worst quarter -- which keeps the question
("how bad is bad") and averages eleven matchups rather than trusting one.

**And the games are not spread evenly.** Most of a flat budget goes on
opponents the party beats comfortably, which the answer does not depend on.
``score`` plays a cheap first pass over the whole field and then spends each
later round on the opponents that decide the number: the ones currently in
the worst quarter, whose rate the CVaR averages, and the ones whose interval
still straddles the quartile boundary, which could enter or leave it.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from typing import Iterator, Sequence

from pkcm.data.dex import Dex, load_dex
from pkcm.engine.legality import Party, ranker_parties
from pkcm.engine.pokemon import PokemonSet
from pkcm.engine.rng import Rng
from pkcm.engine.state import BattleConfig, new_battle
from pkcm.search import MCTS, SearchConfig
from pkcm.search.policy import SearchPolicy, play_out
from pkcm.train.interval import wilson

Team = tuple[PokemonSet, ...]


@dataclass(frozen=True, slots=True)
class FloorConfig:
    """The agent, the field, and how much of the budget to spend where."""

    battle_format: str = "singles"
    regulation: str = "m_b"
    search: SearchConfig = field(default_factory=SearchConfig)
    #: Where the field comes from. ``None`` is the committed archive.
    parties: str | None = None
    #: Field entries to leave out -- a party does not play itself.
    exclude: tuple[int, ...] = ()
    #: Face this many of the field instead of all of it. The seed pass is
    #: every opponent by construction, so against 243 parties it costs 486
    #: battles before any racing begins -- which makes a small budget
    #: meaningless and a generation of a search unaffordable. The subset is
    #: drawn from ``seed`` alone, so every candidate in a run faces the same
    #: opponents and their floors stay comparable.
    opponents: int | None = None
    #: The quantile the floor is taken over. 0.25 is the worst quarter.
    tail: float = 0.25
    #: Games against every opponent before any are singled out. Both seatings
    #: are always played together, so this is rounded up to an even number.
    seed_games: int = 8
    #: Games added per opponent in a round of the race.
    round_games: int = 8
    #: Stop when the tail's own interval is at least this narrow, or when the
    #: budget runs out, whichever comes first.
    target_width: float = 0.06
    seed: int = 700_000


@dataclass(frozen=True, slots=True)
class Matchup:
    opponent: int
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def decided(self) -> int:
        return self.wins + self.losses

    @property
    def rate(self) -> float:
        return self.wins / self.decided if self.decided else 0.5

    def interval(self) -> tuple[float, float]:
        if not self.decided:
            return 0.0, 1.0
        _, low, high = wilson(self.wins, self.decided)
        return low, high

    def __add__(self, other: "Matchup") -> "Matchup":
        return Matchup(self.opponent, self.wins + other.wins,
                       self.losses + other.losses, self.draws + other.draws)


@dataclass(frozen=True, slots=True)
class Selection:
    """Which of the six actually got brought, over all the games played.

    The floor says how bad the bad matchups are. It does not say whether the
    party got there on three Pokemon or on six, and hk's methodology note is
    emphatic that the difference decides games: a slot that only answers one
    archetype is dead weight in every other match, and you are playing three
    against two before the first turn.

    A high floor built on a dead slot reads exactly like a high floor built on
    a flexible six, so this is measured separately rather than folded into the
    number.

    **It measures the picker, not the party.** The three are chosen by the
    agent's team-preview heuristic, which averages over the opponent's six and
    is deliberately optimistic. A rigid-looking party may be a rigid party or
    a rigid picker. Every party in a run is picked by the same heuristic, so
    comparisons hold; the absolute value does not.
    """

    #: Times each registered slot was brought, in team order.
    brought: tuple[int, ...] = ()
    #: The distinct three-Pokemon selections used, with their counts, commonest
    #: first. Truncated -- the tail of one-off selections is not informative.
    combos: tuple[tuple[tuple[int, ...], int], ...] = ()
    battles: int = 0

    @property
    def rates(self) -> tuple[float, ...]:
        if not self.battles:
            return tuple(0.0 for _ in self.brought)
        return tuple(count / self.battles for count in self.brought)

    def live(self, floor: float = 0.15) -> int:
        """How many of the six are brought often enough to be doing work.

        Six of six is a party whose selection genuinely answers the opponent;
        three of six is three Pokemon and three passengers, whatever the
        floor says.
        """
        return sum(1 for rate in self.rates if rate >= floor)

    @property
    def rigidity(self) -> float:
        """Share of battles that used the single commonest three.

        1.0 is a party that brings the same three every game -- which is not
        automatically bad, but it is a different thing from a party that picks
        its three, and the floor cannot tell them apart.
        """
        if not self.battles or not self.combos:
            return 0.0
        return self.combos[0][1] / self.battles


@dataclass(frozen=True, slots=True)
class Floor:
    """What a party is worth, read from the bottom rather than the middle."""

    cvar: float
    mean: float
    minimum: float
    #: A rough interval on the CVaR, from the games behind the tail alone.
    low: float
    high: float
    games: int
    matchups: tuple[Matchup, ...]
    selection: Selection = Selection()

    def worst(self, count: int = 5) -> tuple[Matchup, ...]:
        return tuple(sorted(self.matchups, key=lambda m: m.rate)[:count])


def summarise(matchups: Sequence[Matchup], tail: float,
              selection: Selection | None = None) -> Floor:
    """Fold the per-opponent records into the floor and its neighbours."""
    selection = selection or Selection()
    played = [m for m in matchups if m.decided]
    if not played:
        return Floor(0.5, 0.5, 0.5, 0.0, 1.0, 0, tuple(matchups), selection)
    ordered = sorted(played, key=lambda m: m.rate)
    take = max(1, round(len(ordered) * tail))
    tail_set = ordered[:take]
    wins = sum(m.wins for m in tail_set)
    decided = sum(m.decided for m in tail_set)
    # The interval is over the tail's games pooled. It understates the true
    # uncertainty -- which opponents *are* the tail is itself uncertain -- so
    # it is a floor on the error, not a claim about it.
    _, low, high = wilson(wins, decided) if decided else (0.0, 0.0, 1.0)
    return Floor(
        cvar=wins / decided if decided else 0.5,
        mean=sum(m.rate for m in played) / len(played),
        minimum=ordered[0].rate,
        low=low, high=high,
        games=sum(m.decided + m.draws for m in played),
        matchups=tuple(matchups),
        selection=selection,
    )


def _needs_games(matchups: Sequence[Matchup], tail: float) -> list[int]:
    """Which opponents the next round's games should go to.

    Two kinds. The ones inside the worst quarter, because the CVaR is the
    average of *their* rates and every one of them is in the answer. And the
    ones whose interval still crosses the quartile boundary, because until it
    stops crossing we do not know which set they belong to. Everything else
    is comfortably outside and its exact rate does not enter the number.
    """
    played = [m for m in matchups if m.decided]
    if len(played) < 2:
        return [m.opponent for m in matchups]
    ordered = sorted(played, key=lambda m: m.rate)
    take = max(1, round(len(ordered) * tail))
    boundary = ordered[min(take, len(ordered) - 1)].rate
    wanted = {m.opponent for m in ordered[:take]}
    for m in played:
        low, high = m.interval()
        if low <= boundary <= high:
            wanted.add(m.opponent)
    return sorted(wanted)


# --------------------------------------------------------------------------- #
# Playing
# --------------------------------------------------------------------------- #


def play_pair(dex: Dex, config: FloorConfig, ours: Team, theirs: Team,
              repeat: int) -> tuple[int, int, int, tuple[tuple[int, ...], ...]]:
    """Both seatings of one pairing.

    Returns our wins, losses, draws, and the selections we brought -- one per
    battle, as indices into the registered six. ``SideState.selection`` is
    already the brought order, so this costs nothing beyond reading it off the
    finished state.
    """
    battle_config = BattleConfig(dex=dex, regulation=dex.regulation(config.regulation),
                                 battle_format=config.battle_format)
    seed = config.seed + repeat
    wins = losses = draws = 0
    brought: list[tuple[int, ...]] = []
    for swap in (False, True):
        first = SearchPolicy(MCTS(config.search), Rng.from_seed(seed).cursor())
        second = SearchPolicy(MCTS(config.search), Rng.from_seed(seed + 7777).cursor())
        teams = (theirs, ours) if swap else (ours, theirs)
        state = play_out(new_battle(battle_config, teams, seed=seed), (first, second))
        seat = 1 if swap else 0
        brought.append(tuple(sorted(state.sides[seat].selection)))
        if state.winner is None:
            draws += 1
        elif state.winner == seat:
            wins += 1
        else:
            losses += 1
    return wins, losses, draws, tuple(brought)


# -- worker state, the same shape as matchup's and tournament's ------------- #

_DEX: Dex | None = None
_CONFIG: FloorConfig | None = None
_FIELD: tuple[Party, ...] | None = None
_OURS: Team | None = None


def _start_worker(config: FloorConfig, ours: Team) -> None:
    global _DEX, _CONFIG, _FIELD, _OURS
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:  # pragma: no cover - no network to evaluate
        pass
    _DEX = load_dex()
    _CONFIG = config
    _FIELD = ranker_parties(config.parties)
    _OURS = ours


def _play(task: tuple[int, int]) -> tuple:
    opponent, repeat = task
    wins, losses, draws, brought = play_pair(_DEX, _CONFIG, _OURS,
                                             _FIELD[opponent].team, repeat)
    return opponent, wins, losses, draws, brought


def score(ours: Team, config: FloorConfig, budget: int,
          workers: int | None = None,
          on_round=None) -> Floor:
    """Measure ``ours`` against the field, spending ``budget`` games on it.

    ``budget`` counts battles, not pairings. The first pass takes
    ``seed_games`` against everyone and the rest goes where the answer is.
    ``on_round`` is called with the running ``Floor`` after each round, for a
    caller that wants to print progress.
    """
    from pkcm.train.parallel import default_workers, map_unordered

    field = ranker_parties(config.parties)
    opponents = [i for i in range(len(field)) if i not in set(config.exclude)]
    if config.opponents is not None and config.opponents < len(opponents):
        # Seeded, not per-candidate: two candidates measured against different
        # opponents are not measured against the same question.
        opponents = sorted(random.Random(config.seed).sample(opponents,
                                                             config.opponents))
    records = {i: Matchup(i) for i in opponents}
    # Counted over every battle, not just the tail's: the question "how many
    # of the six are live" is about the whole field, and the tail is where a
    # party is least free to choose.
    slots = [0] * len(ours)
    combos: dict[tuple[int, ...], int] = {}
    battles = 0
    count = workers if workers is not None else default_workers()
    spent = 0
    repeat = 0

    def run(targets: list[int], pairings: int) -> None:
        nonlocal spent, repeat, battles
        tasks = [(opponent, repeat + step)
                 for step in range(pairings) for opponent in targets]
        repeat += pairings
        if count <= 1:
            dex = load_dex()
            results: Iterator = (
                (o, *play_pair(dex, config, ours, field[o].team, r))
                for o, r in tasks)
        else:
            results = map_unordered(_play, tasks, initializer=_start_worker,
                                    initargs=(config, ours), workers=count,
                                    what="pairing")
        for opponent, wins, losses, draws, brought in results:
            records[opponent] += Matchup(opponent, wins, losses, draws)
            spent += wins + losses + draws
            for one in brought:
                battles += 1
                combos[one] = combos.get(one, 0) + 1
                for index in one:
                    slots[index] += 1

    # Each pairing is two battles, so a round of N opponents costs 2N.
    def brought_so_far() -> Selection:
        ranked = sorted(combos.items(), key=lambda pair: -pair[1])[:8]
        return Selection(brought=tuple(slots), combos=tuple(ranked),
                         battles=battles)

    first = max(1, math.ceil(config.seed_games / 2))
    run(opponents, first)
    floor = summarise(list(records.values()), config.tail, brought_so_far())
    if on_round:
        on_round(floor, spent)

    while spent < budget and floor.high - floor.low > config.target_width:
        targets = _needs_games(list(records.values()), config.tail)
        pairings = max(1, math.ceil(config.round_games / 2))
        if spent + 2 * pairings * len(targets) > budget:
            pairings = max(1, (budget - spent) // (2 * max(1, len(targets))))
            if pairings <= 0:
                break
        run(targets, pairings)
        floor = summarise(list(records.values()), config.tail,
                          brought_so_far())
        if on_round:
            on_round(floor, spent)

    return floor
