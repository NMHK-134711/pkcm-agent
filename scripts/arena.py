"""Play policies against each other and report who actually wins.

The only honest measure of a search. A tree that produces confident-looking
statistics and loses to a one-turn damage calculator has learned nothing, and
the statistics will not say so.

Teams are mirrored: both sides play the same pair of teams, once each way, so a
win rate cannot come from having drawn a better team. That halves the variance
before any battles are run.

The interval is Wilson's, not Wald's. Wald collapses to zero when every game
goes the same way -- "0.0% +/- 0.0%" off six games reads as certainty and means
the opposite.

Usage:
    python scripts/arena.py --a greedy --b random --battles 100
    python scripts/arena.py --a search --b greedy --battles 40 --iterations 300
    python scripts/arena.py --a search --b greedy --format doubles --battles 20
    python scripts/arena.py --a net --b search --battles 40 --checkpoint runs/latest/net.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pkcm.data.dex import load_dex  # noqa: E402
from pkcm.engine.legality import random_team  # noqa: E402
from pkcm.engine.rng import Rng  # noqa: E402
from pkcm.engine.state import BattleConfig, new_battle  # noqa: E402
from pkcm.search import MCTS, GreedyPolicy, RandomPolicy, SearchConfig  # noqa: E402
from pkcm.search.policy import SearchPolicy, play_out  # noqa: E402
from pkcm.train.interval import separable, wilson  # noqa: E402


#: One evaluator shared by every search in the run. Loading a checkpoint per
#: battle would cost more than the battles do.
_EVALUATOR = None


def evaluator_for(checkpoint, dex, battle_format: str, trust: float):
    """Load the network once, as the search's prior and leaf value."""
    global _EVALUATOR
    if checkpoint is None:
        return None
    if _EVALUATOR is None:
        from pkcm.envs.encoding import SCALAR_SIZE, action_space_size
        from pkcm.train.evaluator import from_checkpoint

        registered, brought = dex.regulation("m_b").bring_select(battle_format)
        _EVALUATOR = from_checkpoint(
            checkpoint, dex, action_space_size(registered, brought),
            SCALAR_SIZE, device="cpu", trust=trust)
    return _EVALUATOR


def build(name: str, seed: int, iterations: int, determinizations: int,
          rollout: int, prior: float | None = None, evaluator=None,
          ablate: tuple[str, ...] = (), exploration: float | None = None):
    """``ablate`` switches named ``SearchConfig`` flags off.

    An ablation is the only way to find out whether a change did anything. Two
    runs that differ in one flag and nothing else, mirrored teams, both
    seatings: that is a measurement. A number on its own is not.
    """
    if name == "random":
        return RandomPolicy.seeded(seed)
    if name == "greedy":
        return GreedyPolicy.seeded(seed)
    if name == "search":
        extra = {} if prior is None else {"prior_weight": prior}
        extra.update({flag: False for flag in ablate})
        if exploration is not None:
            extra["exploration"] = exploration
        config = SearchConfig(iterations=iterations,
                              determinizations=determinizations,
                              rollout_turns=rollout, **extra)
        return SearchPolicy(MCTS(config, evaluator=evaluator),
                            Rng.from_seed(seed).cursor())
    if name == "net":
        # Same search, network prior and value instead of the handcrafted ones.
        # A separate name so a run can put the two against each other, which is
        # the only comparison that says whether training did anything.
        if evaluator is None:
            raise SystemExit("--checkpoint is required for the 'net' policy")
        config = SearchConfig(iterations=iterations,
                              determinizations=determinizations,
                              rollout_turns=rollout)
        return SearchPolicy(MCTS(config, evaluator=evaluator),
                            Rng.from_seed(seed).cursor())
    raise SystemExit(f"unknown policy {name!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", default="search", help="random | greedy | search")
    parser.add_argument("--b", default="greedy")
    parser.add_argument("--battles", type=int, default=40)
    parser.add_argument("--format", default="singles", choices=("singles", "doubles"))
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--determinizations", type=int, default=15)
    parser.add_argument("--rollout", type=int, default=0,
                        help="turns to play out at a leaf; 0 uses the heuristic")
    parser.add_argument("--prior", type=float, default=None,
                        help="PUCT prior weight; 0 ablates the prior entirely")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="network to use for the 'net' policy")
    parser.add_argument("--exploration", type=float, default=None,
                        help="UCB1 term for --a, on top of PUCT's prior term")
    parser.add_argument("--exploration-b", type=float, default=None,
                        help="the same for --b, so the two can be matched "
                             "directly. Two win rates against a third party "
                             "waste most of their samples: beating greedy 75%% "
                             "and 69%% leaves intervals that overlap for a "
                             "hundred games, where the head to head does not")
    parser.add_argument("--ablate", default="",
                        help="SearchConfig flags to switch off in --a, comma "
                             "separated (e.g. normalize_value,sample_opponent)")
    parser.add_argument("--trust", type=float, default=1.0,
                        help="how far 'net' believes the network over the heuristic")
    args = parser.parse_args()

    dex = load_dex()
    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"),
                          battle_format=args.format)

    ablated = tuple(flag for flag in args.ablate.split(",") if flag)
    for flag in ablated:
        if not hasattr(SearchConfig(), flag):
            raise SystemExit(f"SearchConfig has no flag {flag!r}")
    evaluator = evaluator_for(args.checkpoint, dex, args.format, args.trust)
    wins = {"a": 0, "b": 0, "draw": 0}
    start = time.perf_counter()
    for match in range(args.battles):
        teams = tuple(
            random_team(dex, config.regulation,
                        Rng.from_seed(args.seed + match * 2 + offset).cursor(),
                        args.format)
            for offset in (1, 2)
        )
        # Same teams, both seatings. A win rate from the draw is not a win rate.
        for swap in (False, True):
            a = build(args.a, args.seed + match, args.iterations,
                      args.determinizations, args.rollout, args.prior, evaluator,
                      ablate=ablated, exploration=args.exploration)
            b = build(args.b, args.seed + match + 5000, args.iterations,
                      args.determinizations, args.rollout, args.prior, evaluator,
                      exploration=args.exploration_b)
            policies = (b, a) if swap else (a, b)
            state = new_battle(config, teams, seed=args.seed + match)
            state = play_out(state, policies)

            a_side = 1 if swap else 0
            if state.winner is None:
                wins["draw"] += 1
            elif state.winner == a_side:
                wins["a"] += 1
            else:
                wins["b"] += 1

    elapsed = time.perf_counter() - start
    played = wins["a"] + wins["b"] + wins["draw"]
    decided = wins["a"] + wins["b"]
    rate, low, high = wilson(wins["a"], decided)

    print(f"{args.a} vs {args.b}   {args.format}   {played} battles "
          f"({elapsed:.1f}s, {played / elapsed:.1f}/s)")
    print(f"  {args.a:8} {wins['a']:4}")
    print(f"  {args.b:8} {wins['b']:4}")
    print(f"  draw     {wins['draw']:4}")
    print(f"  win rate {rate:.1%}   95% [{low:.1%}, {high:.1%}]")
    if not separable(wins["a"], decided):
        print("  -> not separable from a coin flip at this sample size")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
