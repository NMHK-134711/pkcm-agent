"""Two trained networks against each other, each piloting the party it learned.

    python scripts/head_to_head.py runs/pilot43/best.pt runs/curriculum4/best.pt \
        --party-a 43 --party-b 7 --matches 100

``judge.py`` cannot ask this. It swaps the seats between the two games of a
match while holding the teams to the seats, which is exactly right when both
sides play the same distribution -- and exactly wrong here, because on the swap
each network would inherit the other's party. A specialist measured piloting
somebody else's team is not the thing being asked about.

So the pairing is glued: A always brings its party, B always brings its own,
and the two games of a match differ only in who occupies seat zero. That keeps
the seat bias out without letting the teams move.

**This does not isolate skill.** The parties are not equally strong -- the
forty-six party round robin spreads from 66.0% down to 34.5% -- so a win here
is the pair, network and team together, which is what actually gets deployed.
To ask which network is better at a *fixed* team, hand both the same one
(``--party-b`` equal to ``--party-a``) and the difference is skill alone.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pkcm.data.dex import load_dex  # noqa: E402
from pkcm.engine.legality import ranker_parties  # noqa: E402
from pkcm.engine.rng import Rng  # noqa: E402
from pkcm.engine.state import BattleConfig, new_battle  # noqa: E402
from pkcm.search import MCTS, SearchConfig  # noqa: E402
from pkcm.search.policy import SearchPolicy, play_out  # noqa: E402
from pkcm.train.interval import wilson  # noqa: E402
from pkcm.train.matchup import MatchConfig, Record, _evaluator  # noqa: E402
from pkcm.train.parallel import default_workers, map_unordered  # noqa: E402


@dataclass(frozen=True)
class Pairing:
    """Which network brings which party, and how hard each of them thinks."""

    checkpoint_a: str
    checkpoint_b: str
    party_a: int
    party_b: int
    iterations: int
    battle_format: str = "singles"
    regulation: str = "m_b"


_DEX = None
_CONFIG: "Pairing | None" = None


def _start_worker(config: Pairing) -> None:
    global _DEX, _CONFIG
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        import torch

        # One process per core already saturates the machine, and the thread
        # count is not only a speed choice: a CPU matmul reduces in a different
        # order on many threads than on one, and an argmax that flips there
        # sends the battle somewhere else.
        torch.set_num_threads(1)
    except ImportError:  # pragma: no cover - torch is needed for a checkpoint
        pass
    _DEX = load_dex()
    _CONFIG = config


def _play(match: int) -> Record:
    assert _DEX is not None and _CONFIG is not None, "worker was not initialised"
    return play(_DEX, _CONFIG, match)


def play(dex, config: Pairing, match: int) -> Record:
    """One match: the same pair of teams, once from each seat.

    Returns A's record. The seeds come from ``match`` alone so the pool can
    hand matches out in any order and still be reproducible.
    """
    battle_config = BattleConfig(dex=dex,
                                 regulation=dex.regulation(config.regulation),
                                 battle_format=config.battle_format)
    parties = ranker_parties()
    search = SearchConfig(iterations=config.iterations,
                          determinizations=max(4, config.iterations // 40),
                          leaf_batch=16)

    # ``_evaluator`` caches per path, so both networks live in one worker.
    shape = MatchConfig(battle_format=config.battle_format,
                        regulation=config.regulation, trust=1.0)
    evaluator_a = _evaluator(dex, shape, config.checkpoint_a)
    evaluator_b = _evaluator(dex, shape, config.checkpoint_b)

    record = Record()
    for swap in (False, True):
        # The teams travel with their networks; only the seats change.
        teams = ((parties[config.party_b].team, parties[config.party_a].team)
                 if swap else
                 (parties[config.party_a].team, parties[config.party_b].team))
        first = SearchPolicy(MCTS(search, evaluator=evaluator_a),
                             Rng.from_seed(match).cursor())
        second = SearchPolicy(MCTS(search, evaluator=evaluator_b),
                              Rng.from_seed(match + 7777).cursor())
        policies = (second, first) if swap else (first, second)
        state = play_out(new_battle(battle_config, teams, seed=match), policies)

        seat_a = 1 if swap else 0
        if state.winner is None:
            record += Record(draws=1)
        elif state.winner == seat_a:
            record += Record(wins=1)
        else:
            record += Record(losses=1)
    return record


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(
        allow_abbrev=False, description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint_a")
    parser.add_argument("checkpoint_b")
    parser.add_argument("--party-a", type=int, required=True)
    parser.add_argument("--party-b", type=int, required=True)
    parser.add_argument("--matches", type=int, default=100,
                        help="each is two games, one per seat")
    parser.add_argument("--search-iterations", type=int, default=800)
    parser.add_argument("--format", default="singles", choices=("singles", "doubles"))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    parties = ranker_parties()
    workers = args.workers or default_workers()
    config = Pairing(checkpoint_a=args.checkpoint_a, checkpoint_b=args.checkpoint_b,
                     party_a=args.party_a, party_b=args.party_b,
                     iterations=args.search_iterations, battle_format=args.format)

    print(f"A: {Path(args.checkpoint_a).as_posix()} with party {args.party_a} "
          f"-- {parties[args.party_a].title[:40]}")
    print(f"B: {Path(args.checkpoint_b).as_posix()} with party {args.party_b} "
          f"-- {parties[args.party_b].title[:40]}")
    print(f"{args.matches} matches ({2 * args.matches} games) | "
          f"{args.search_iterations} sims | {workers} workers", flush=True)

    total = Record()
    started = beat = time.perf_counter()
    for done, one in enumerate(
            map_unordered(_play, range(args.matches), initializer=_start_worker,
                          initargs=(config,), workers=workers, what="match"), 1):
        total += one
        now = time.perf_counter()
        if now - beat >= 30.0 or done == args.matches:
            left = (args.matches - done) / max(done / (now - started), 1e-9)
            print(f"  {done}/{args.matches}  {total.wins}-{total.losses}  "
                  f"~{left:.0f}s left", flush=True)
            beat = now

    rate = total.wins / max(1, total.decided)
    low, high = wilson(total.wins, max(1, total.decided))[1:]
    print(f"\n  {total.wins}-{total.losses} ({total.draws} drawn) over "
          f"{total.decided} decided games")
    print(f"  A wins {rate:.1%} [{low:.1%}, {high:.1%}]")
    if low > 0.5:
        print("  A is stronger, separably.")
    elif high < 0.5:
        print("  B is stronger, separably.")
    else:
        print("  not separable.")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "checkpoint_a": args.checkpoint_a, "party_a": args.party_a,
            "checkpoint_b": args.checkpoint_b, "party_b": args.party_b,
            "iterations": args.search_iterations,
            "wins": total.wins, "losses": total.losses, "draws": total.draws,
            "rate": rate, "low": low, "high": high,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
