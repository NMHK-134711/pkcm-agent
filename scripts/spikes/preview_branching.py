"""Does capping the team preview at 24 of its 120 orderings cost games?

``runs/preview_truncation.json`` says the cap keeps the wrong ones: over 40
positions, only 15.3% of a wide search's visits landed on the 24 that
``joint_actions`` would have kept, against a 20% baseline -- and the PUCT
prior is built from the same scores, so it was pulling *towards* those 24 the
whole time. That measures the search against its own value estimates. This
measures it against games.

Nothing in the search changes. In singles the cap binds at the team preview
and nowhere else -- a battle node offers at most ten options and a forced
switch two -- so ``max_branching=120`` alters exactly one decision per game,
which is what makes this a clean comparison rather than two agents.

    python scripts/spikes/preview_branching.py --matches 600
    python scripts/spikes/preview_branching.py --matches 200 --iterations 3200

Teams are drawn from the ranker pool rather than fixed, because a cap that
helps one party and hurts another averages to nothing on a mirror of a single
team. Both seatings of every pair, as everywhere else here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pkcm.search import SearchConfig  # noqa: E402
from pkcm.train.interval import wilson  # noqa: E402
from pkcm.train.matchup import MatchConfig, Record, stream  # noqa: E402
from pkcm.train.parallel import default_workers  # noqa: E402


def main() -> int:
    for out in (sys.stdout, sys.stderr):
        try:
            out.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matches", type=int, default=600,
                        help="pairs of teams; each is played both ways round, "
                             "so games are twice this")
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--branching", type=int, default=120,
                        help="the uncapped side's limit. 120 is every preview "
                             "ordering in singles")
    parser.add_argument("--teams", default="ranker")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    wide = SearchConfig(iterations=args.iterations, max_branching=args.branching)
    narrow = SearchConfig(iterations=args.iterations)
    config = MatchConfig(checkpoint=None, teams=args.teams,
                         search=wide, search_b=narrow)
    workers = args.workers if args.workers is not None else default_workers()
    out = Path(args.out) if args.out else \
        ROOT / f"runs/preview_branching_{args.iterations}.json"

    print(f"uncapped preview ({args.branching}) vs the current cap "
          f"({narrow.max_branching}), {args.iterations} simulations a side")
    print(f"{args.matches} pairs = {2 * args.matches} games on {workers} workers, "
          f"teams from {args.teams}")
    total = Record()
    started = time.time()
    for done, record in enumerate(stream(config, args.matches, workers), 1):
        total += record
        if done % 25 == 0 or done == args.matches:
            rate, low, high = wilson(total.wins, total.decided)
            spent = time.time() - started
            left = spent / done * (args.matches - done)
            print(f"  {done:4}/{args.matches}  {total.wins}-{total.losses}"
                  f"-{total.draws}  {rate:.1%} [{low:.1%}, {high:.1%}]  "
                  f"{spent / 60:.0f}m spent, {left / 60:.0f}m left", flush=True)

    rate, low, high = wilson(total.wins, total.decided)
    separable = low > 0.5 or high < 0.5
    summary = {
        "iterations": args.iterations,
        "branching_wide": args.branching,
        "branching_narrow": narrow.max_branching,
        "teams": args.teams,
        "matches": args.matches,
        "wins": total.wins, "losses": total.losses, "draws": total.draws,
        "rate": round(rate, 4), "low": round(low, 4), "high": round(high, 4),
        "separable": separable,
        "minutes": round((time.time() - started) / 60, 1),
    }
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print()
    print(f"uncapped preview: {total.wins}-{total.losses}-{total.draws}  "
          f"{rate:.1%} [{low:.1%}, {high:.1%}]")
    print("separable from even" if separable else
          "NOT separable from even -- the interval covers 50%")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
