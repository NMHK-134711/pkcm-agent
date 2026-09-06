"""Rank parties by their floor rather than by their average.

    python scripts/party_floor.py --candidates 8 --search-iterations 3200
    python scripts/party_floor.py --entrants 42,43,39 --budget 4000

The field does not sort -- 43 beats 14 beats 42 beats 43, every edge
separable (``runs/tournament_contenders.json``) -- so a mean win rate mostly
reports which opponents were in the field. What survives the cycles is
whether a party has an answer to most of what it meets, and with three of six
brought, an answer can be a different three rather than a different party.

The number is the CVaR of the worst quarter of matchups, not the minimum:
over forty-five opponents the sample minimum is the noisiest statistic
available, and optimising it chases the unluckiest matchup rather than the
genuinely bad one. See ``pkcm.train.party_floor``.

Games are not spread evenly. Each round goes to the pairings that decide the
answer -- the ones inside somebody's worst quarter, and the ones whose
interval still straddles their quartile boundary. Pairings wanted by two
candidates are played once and counted for both, which is most of the saving
when several candidates are scored together.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pkcm.engine.legality import ranker_parties  # noqa: E402
from pkcm.search import SearchConfig  # noqa: E402
from pkcm.train.parallel import default_workers  # noqa: E402
from pkcm.train.party_floor import Matchup, summarise  # noqa: E402
from pkcm.train.party_floor import _needs_games as needs_games  # noqa: E402
from pkcm.train.tournament import TournamentConfig, stream  # noqa: E402


def main() -> int:
    for out in (sys.stdout, sys.stderr):
        try:
            out.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entrants", default=None,
                        help="comma-separated party indices to score. Default "
                             "takes the best --candidates by --shortlist")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--field",
                        default=str(ROOT / "data" / "champions" / "parties_field.json"),
                        help="the parties to be scored against, and the ones "
                             "--entrants indexes into. The 253-party field "
                             "covers 98.5% of the ladder's slots against the "
                             "46-party archive's 82.3%")
    parser.add_argument("--shortlist", default="runs/tournament_46_v3.json",
                        help="an earlier round robin, used only to choose who "
                             "is worth scoring. Its games are not reused: it "
                             "was played by a different agent")
    parser.add_argument("--search-iterations", type=int, default=3200)
    parser.add_argument("--preview-iterations", type=int, default=None,
                        help="defaults to four times --search-iterations, "
                             "which is the agent as deployed")
    parser.add_argument("--seed-games", type=int, default=8,
                        help="games against every opponent before any are "
                             "singled out")
    parser.add_argument("--round-games", type=int, default=8)
    parser.add_argument("--budget", type=int, default=12000,
                        help="battles, in total, across every candidate")
    parser.add_argument("--tail", type=float, default=0.25)
    parser.add_argument("--target-width", type=float, default=0.08)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", default=str(ROOT / "runs/party_floor.json"))
    args = parser.parse_args()

    field = ranker_parties(args.field)
    if args.entrants:
        candidates = [int(one) for one in args.entrants.split(",")]
    else:
        shortlist = json.loads(Path(args.shortlist).read_text(encoding="utf-8"))
        prior = defaultdict(lambda: [0, 0, 0])
        for fixture in shortlist["fixtures"]:
            row = prior[(fixture["a"], fixture["b"])]
            row[0] += fixture["a_wins"]
            row[1] += fixture["b_wins"]
        seen = sorted({p for pair in prior for p in pair})

        def prior_records(party):
            out = []
            for other in seen:
                if other == party:
                    continue
                if (party, other) in prior:
                    wins, losses, _ = prior[(party, other)]
                else:
                    losses, wins, _ = prior[(other, party)]
                out.append(Matchup(other, wins, losses))
            return out

        ranked = sorted(seen, key=lambda p: -summarise(prior_records(p), args.tail).cvar)
        candidates = ranked[:args.candidates]

    preview = args.preview_iterations
    if preview is None:
        preview = args.search_iterations * 4
    config = TournamentConfig(
        parties=args.field,
        search=SearchConfig(iterations=args.search_iterations,
                            preview_iterations=preview,
                            determinizations=max(4, args.search_iterations // 20)),
        seed=700_000)
    workers = args.workers if args.workers is not None else default_workers()

    print(f"scoring {len(candidates)} parties against a field of {len(field)}")
    print(f"candidates: {', '.join(str(one) for one in candidates)}")
    print(f"{args.search_iterations} simulations, preview {preview}, "
          f"budget {args.budget} battles on {workers} workers")

    # records[candidate][opponent]
    records: dict[int, dict[int, Matchup]] = {
        one: {other: Matchup(other) for other in range(len(field)) if other != one}
        for one in candidates
    }
    played: dict[tuple[int, int], int] = defaultdict(int)
    spent = 0
    started = time.time()

    def run(pairs: set[tuple[int, int]], pairings: int) -> None:
        """Play ``pairings`` more fixtures of each pair, both seatings each."""
        nonlocal spent
        schedule = []
        for a, b in sorted(pairs):
            for step in range(pairings):
                schedule.append((a, b, played[(a, b)] + step))
            played[(a, b)] += pairings
        for result in stream(config, schedule, workers):
            spent += result.a_wins + result.b_wins + result.draws
            for who, mine, theirs in ((result.a, result.a_wins, result.b_wins),
                                      (result.b, result.b_wins, result.a_wins)):
                other = result.b if who == result.a else result.a
                if who in records and other in records[who]:
                    records[who][other] += Matchup(other, mine, theirs, result.draws)

    def floors():
        return {one: summarise(list(rows.values()), args.tail)
                for one, rows in records.items()}

    def report(label: str) -> None:
        spread = floors()
        print(f"\n  {label}: {spent} battles, {(time.time() - started) / 60:.0f}m")
        for one, floor in sorted(spread.items(), key=lambda kv: -kv[1].cvar):
            worst = ", ".join(f"{m.opponent}@{m.rate:.0%}" for m in floor.worst(3))
            print(f"    {one:3} CVaR {floor.cvar:5.1%} [{floor.low:.1%}, {floor.high:.1%}]"
                  f"  mean {floor.mean:5.1%}  worst {worst}")

    everyone = {tuple(sorted((one, other)))
                for one in candidates for other in records[one]}
    run(everyone, max(1, args.seed_games // 2))
    report("first pass")

    while spent < args.budget:
        if all(f.high - f.low <= args.target_width for f in floors().values()):
            print("\n  every candidate is inside the target width")
            break
        # Every candidate keeps getting games, the ones already inside the
        # width included. The tail is picked by the same games that score it,
        # so its bias points down and lifts as they accumulate -- stopping one
        # candidate early would leave it compared at a different bias from the
        # rest, which is not a comparison.
        wanted: set[tuple[int, int]] = set()
        for one, rows in records.items():
            for other in needs_games(list(rows.values()), args.tail):
                wanted.add(tuple(sorted((one, other))))
        if not wanted:
            break
        pairings = max(1, args.round_games // 2)
        if spent + 2 * pairings * len(wanted) > args.budget:
            pairings = max(0, (args.budget - spent) // (2 * max(1, len(wanted))))
            if pairings <= 0:
                break
        run(wanted, pairings)
        report(f"after {len(wanted)} pairings")

    spread = floors()
    payload = {
        "search_iterations": args.search_iterations,
        "preview_iterations": preview,
        "tail": args.tail,
        "battles": spent,
        "minutes": round((time.time() - started) / 60, 1),
        "parties": [
            {"party": one, "title": field[one].title,
             "cvar": round(floor.cvar, 4), "low": round(floor.low, 4),
             "high": round(floor.high, 4), "mean": round(floor.mean, 4),
             "minimum": round(floor.minimum, 4), "games": floor.games,
             # The tail's bias depends on how many games are behind it, so
             # this belongs next to the CVaR rather than being trusted apart
             # from it.
             "tail_games": sum(m.decided for m in floor.worst(
                 max(1, round(len(floor.matchups) * args.tail)))),
             "worst": [{"opponent": m.opponent, "rate": round(m.rate, 4),
                        "games": m.decided} for m in floor.worst(6)]}
            for one, floor in sorted(spread.items(), key=lambda kv: -kv[1].cvar)
        ],
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
