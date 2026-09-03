"""Choose a curriculum party set from a round robin -- by balance, not by rank.

The overnight picker of 2026-09-02 took the top four by win rate and chose
39, 43, 14, 42. Party 39 beat 43 eight games to none and 14 seven to one in
that same round robin, so gate games between two networks -- both sides
drawing from the four -- were decided by who drew 39, and six curriculum
runs went 0/8 while curriculum4's parties promoted three times on the same
engine. A curriculum needs matchups that a *move* can turn, so the rule here
is: among the strongest K parties, the four whose internal matchups sit
closest to 50-50, with no pair allowed past a lopsided ceiling.

    python scripts/pick_curriculum.py runs/tournament_46_v2.json
    python scripts/pick_curriculum.py runs/tournament_46_v2.json --top 10 --size 4 --max-pair 0.3

Prints a table of the best sets and the chosen one's six pair records, so
the reason is on the page next to the answer. Reads only the fixtures in the
tournament JSON; eight games per pair is coarse, which is why the ceiling is
a hard filter and the mean is only used to order what survives.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path


def pair_records(fixtures: list[dict]) -> dict[tuple[int, int], list[int]]:
    """``(a, b) -> [a_wins, b_wins]`` over every fixture, both orientations."""
    records: dict[tuple[int, int], list[int]] = {}
    for fixture in fixtures:
        a, b = fixture["a"], fixture["b"]
        records.setdefault((a, b), [0, 0])
        records.setdefault((b, a), [0, 0])
        records[(a, b)][0] += fixture["a_wins"]
        records[(a, b)][1] += fixture["b_wins"]
        records[(b, a)][0] += fixture["b_wins"]
        records[(b, a)][1] += fixture["a_wins"]
    return records


def bias(parties: tuple[int, ...], records) -> tuple[float, float]:
    """Mean and max of |p - 0.5| over the set's internal pairs."""
    deviations = []
    for a, b in itertools.combinations(parties, 2):
        wins, losses = records.get((a, b), (0, 0))
        games = wins + losses
        deviations.append(abs(wins / games - 0.5) if games else 0.0)
    return sum(deviations) / len(deviations), max(deviations)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tournament", type=Path, help="scripts/tournament.py output")
    parser.add_argument("--top", type=int, default=10,
                        help="only parties ranked this high are eligible")
    parser.add_argument("--size", type=int, default=4, help="parties per set")
    parser.add_argument("--max-pair", type=float, default=0.3,
                        help="reject a set if any internal pair deviates more "
                             "than this from 50%% (0.3 = worse than 6-2 of 8)")
    parser.add_argument("--show", type=int, default=8, help="how many sets to list")
    args = parser.parse_args()

    data = json.loads(args.tournament.read_text(encoding="utf-8"))
    standings = data["standings"]
    rank = {row["party"]: i + 1 for i, row in enumerate(standings)}
    rate = {row["party"]: row["rate"] for row in standings}
    title = {row["party"]: row["title"] for row in standings}
    records = pair_records(data["fixtures"])

    eligible = [row["party"] for row in standings[: args.top]]
    scored = []
    for parties in itertools.combinations(eligible, args.size):
        mean, worst = bias(parties, records)
        if worst > args.max_pair:
            continue
        strength = sum(rate[p] for p in parties) / len(parties)
        scored.append((mean, worst, -strength, parties))
    scored.sort()

    print(f"{len(eligible)} eligible parties (top {args.top}), "
          f"{len(scored)} sets of {args.size} under the pair ceiling {args.max_pair}\n")
    print(f"{'set':<20} {'mean|p-.5|':>10} {'max':>6} {'mean rate':>10}  ranks")
    for mean, worst, neg_strength, parties in scored[: args.show]:
        label = ",".join(str(p) for p in parties)
        print(f"{label:<20} {mean:>10.3f} {worst:>6.3f} {-neg_strength:>10.1%}  "
              f"{[rank[p] for p in parties]}")

    if not scored:
        print("\nno set survives the ceiling -- raise --max-pair or --top")
        return 1
    mean, worst, _, chosen = scored[0]
    print(f"\nchosen: --teams parties:{','.join(str(p) for p in chosen)}")
    for p in chosen:
        print(f"  {p:2}  #{rank[p]:<2} {rate[p]:.1%}  {title[p][:40]}")
    print("  internal pairs:")
    for a, b in itertools.combinations(chosen, 2):
        wins, losses = records[(a, b)]
        print(f"    {a:2} vs {b:2}: {wins}-{losses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
