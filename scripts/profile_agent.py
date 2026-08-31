"""One network against the handcrafted search, party by party.

    python scripts/profile_agent.py runs/curriculum4/best.pt --repeats 30

Both sides are handed the *same* party, and each match plays it from both
seats. That is the point: a party's own strength cancels, so what is left is
which agent gets more out of it. Comparing an agent's favourite party against
another agent's favourite would price the parties -- the round robin already
measured a 22-point spread between them -- and call it a difference in skill.

Repeating over every imported party is also how the game's randomness is
handled. One party is a few dozen coin flips; twenty parties from both seats is
a profile, and the per-party rows say where a network's strength actually
lives rather than averaging it into a single number that hides the shape.

Parties the network never trained on are the interesting rows. A curriculum
that only lifts its own four has taught the network those four.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pkcm.engine.legality import ranker_parties  # noqa: E402
from pkcm.search import SearchConfig  # noqa: E402
from pkcm.train.interval import wilson  # noqa: E402
from pkcm.train.matchup import MatchConfig, Record  # noqa: E402
from pkcm.train.matchup import stream as play  # noqa: E402
from pkcm.train.parallel import default_workers  # noqa: E402


def main() -> int:
    for out in (sys.stdout, sys.stderr):
        try:
            out.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(
        allow_abbrev=False, description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--repeats", type=int, default=30,
                        help="matches per party; each is two games, one per seat")
    parser.add_argument("--parties", default=None,
                        help="comma separated indices, default every imported party")
    parser.add_argument("--trained-on", default="",
                        help="comma separated indices the network was trained "
                             "on, marked in the table so held-out rows are "
                             "readable at a glance")
    parser.add_argument("--search-iterations", type=int, default=800)
    parser.add_argument("--format", default="singles", choices=("singles", "doubles"))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    parties = ranker_parties()
    picked = ([int(one) for one in args.parties.split(",")] if args.parties
              else list(range(len(parties))))
    trained = {int(one) for one in args.trained_on.split(",") if one.strip()}
    workers = args.workers or default_workers()
    search = SearchConfig(iterations=args.search_iterations,
                          determinizations=max(4, args.search_iterations // 40),
                          leaf_batch=16)

    print(f"{Path(args.checkpoint).as_posix()} vs the handcrafted search | "
          f"{len(picked)} parties | {args.repeats} matches each "
          f"({2 * args.repeats} games) | {args.search_iterations} sims | "
          f"{workers} workers", flush=True)

    rows = []
    started = time.perf_counter()
    for done, party in enumerate(picked, 1):
        config = MatchConfig(checkpoint=args.checkpoint,
                             battle_format=args.format,
                             teams=f"parties:{party}", search=search, trust=1.0)
        tally = Record()
        for one in play(config, args.repeats, workers):
            tally += one
        rate = tally.wins / max(1, tally.decided)
        low, high = wilson(tally.wins, max(1, tally.decided))[1:]
        rows.append({"party": party, "title": parties[party].title,
                     "wins": tally.wins, "losses": tally.losses,
                     "draws": tally.draws, "rate": rate,
                     "low": low, "high": high,
                     "trained_on": party in trained})
        spent = time.perf_counter() - started
        left = spent / done * (len(picked) - done)
        print(f"  party {party:>2} {'*' if party in trained else ' '} "
              f"{rate:6.1%} [{low:.1%}, {high:.1%}]  "
              f"{tally.wins}-{tally.losses}  ~{left:.0f}s left", flush=True)

    rows.sort(key=lambda row: row["rate"], reverse=True)
    wins = sum(row["wins"] for row in rows)
    decided = sum(row["wins"] + row["losses"] for row in rows)
    overall, low, high = (wins / max(1, decided), *wilson(wins, max(1, decided))[1:])

    width = max(len(row["title"]) for row in rows)
    print(f"\n  {'#':>2}  {'party':>5}  {'win rate':>21}  {'W-L':>9}  team")
    for place, row in enumerate(rows, 1):
        mark = "*" if row["trained_on"] else " "
        span = f"{row['rate']:6.1%} [{row['low']:.1%}, {row['high']:.1%}]"
        print(f"  {place:>2}{mark} {row['party']:>5}  {span:>21}  "
              f"{row['wins']:>4}-{row['losses']:<4}  {row['title'][:width]}")
    if trained:
        print("\n  * trained on")
        for label, group in (("trained parties", True), ("held out", False)):
            part = [row for row in rows if row["trained_on"] is group]
            if not part:
                continue
            w = sum(row["wins"] for row in part)
            d = sum(row["wins"] + row["losses"] for row in part)
            lo, hi = wilson(w, max(1, d))[1:]
            print(f"  {label:>16}: {w / max(1, d):6.1%} [{lo:.1%}, {hi:.1%}] "
                  f"over {d} games, {len(part)} parties")
    print(f"\n  overall {overall:.1%} [{low:.1%}, {high:.1%}] over {decided} games")
    print("  separably stronger than the handcrafted search."
          if low > 0.5 else
          "  not separable from the handcrafted search.")

    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(
            {"checkpoint": args.checkpoint, "repeats": args.repeats,
             "iterations": args.search_iterations,
             "overall": {"wins": wins, "decided": decided, "rate": overall,
                         "low": low, "high": high},
             "parties": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  written to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
