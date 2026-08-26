"""Play policies against each other and report who actually wins.

The only honest measure of a search. A tree that produces confident-looking
statistics and loses to a one-turn damage calculator has learned nothing, and
the statistics will not say so.

Teams are mirrored: both sides play the same pair of teams, once each way, so a
win rate cannot come from having drawn a better team. That halves the variance
before any battles are run.

Usage:
    python scripts/arena.py --a greedy --b random --battles 100
    python scripts/arena.py --a search --b greedy --battles 40 --iterations 300
    python scripts/arena.py --a search --b greedy --format doubles --battles 20
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


def build(name: str, seed: int, iterations: int, determinizations: int,
          rollout: int):
    if name == "random":
        return RandomPolicy.seeded(seed)
    if name == "greedy":
        return GreedyPolicy.seeded(seed)
    if name == "search":
        config = SearchConfig(iterations=iterations,
                              determinizations=determinizations,
                              rollout_turns=rollout)
        return SearchPolicy(MCTS(config), Rng.from_seed(seed).cursor())
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
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dex = load_dex()
    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"),
                          battle_format=args.format)

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
                      args.determinizations, args.rollout)
            b = build(args.b, args.seed + match + 5000, args.iterations,
                      args.determinizations, args.rollout)
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
    rate = wins["a"] / decided if decided else 0.0
    # Wald interval. Rough, but enough to say whether a gap is worth believing.
    margin = 1.96 * (rate * (1 - rate) / decided) ** 0.5 if decided else 0.0

    print(f"{args.a} vs {args.b}   {args.format}   {played} battles "
          f"({elapsed:.1f}s, {played / elapsed:.1f}/s)")
    print(f"  {args.a:8} {wins['a']:4}")
    print(f"  {args.b:8} {wins['b']:4}")
    print(f"  draw     {wins['draw']:4}")
    print(f"  win rate {rate:.1%}  +/- {margin:.1%}")
    if decided and abs(rate - 0.5) < margin:
        print("  -> not separable at this sample size")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
