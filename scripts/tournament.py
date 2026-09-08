"""Which of the imported parties wins most, with the agent driving both sides.

    python scripts/tournament.py --repeats 6
    python scripts/tournament.py --entrants 1,8,13 --repeats 40 \
        --search-iterations 3200

Two stages are the intended use. The first is a full round robin at the
self-play iteration count, which is cheap enough to run every entrant against
every other and is only asked to say which few are worth a closer look. The
second re-plays those few at ``DEPLOY_ITERATIONS`` -- the strength the agent
actually plays at -- because a team that needs the search to find its line is
not the same team at 800 simulations as at 3200.

The result is written to JSON so the second stage does not have to re-derive
the field by eye.
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
from pkcm.train.parallel import default_workers  # noqa: E402
from pkcm.train.tournament import (  # noqa: E402
    Result, TournamentConfig, fixtures, standings, stream,
)


def main() -> int:
    # Team titles are whatever the ranker typed -- Korean, Japanese, emoji --
    # and a Windows console defaults to cp949, which raises on the first
    # character it cannot encode. That is a display problem, and it once threw
    # away three hours of finished games, so it is downgraded to a replacement
    # character here.
    for stream_out in (sys.stdout, sys.stderr):
        try:
            stream_out.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--opponents", type=int, default=None,
                        help="play about this many of the other entrants "
                             "instead of all of them. The 253-party run "
                             "resolves the first party only from the ninth, "
                             "so a full field buys a shortlist rather than a "
                             "ranking; this buys the same shortlist for less. "
                             "The subgraph is seeded, so two machines sharing "
                             "one --out draw the same one")
    parser.add_argument("--repeats", type=int, default=6,
                        help="battles per pair per seating. Every entrant "
                             "plays 2 x repeats x (field - 1) games")
    parser.add_argument("--entrants", default=None,
                        help="comma-separated party indices. Default is all "
                             "of them")
    parser.add_argument("--parties", default=None,
                        help="a party file to run instead of the committed "
                             "archive. data/champions/parties_field.json is "
                             "the 253 built from the hand parties and the "
                             "pokesol imports, covering 98.5%% of the ladder's "
                             "slots against the archive's 82.3%%")
    parser.add_argument("--search-iterations", type=int, default=800)
    parser.add_argument("--evaluation", default="material",
                        choices=("material", "pressure", "blind"),
                        help="the leaf value. \"blind\" is the ablation: no "
                             "opinion about a position except at terminals, "
                             "so what remains is the prior and the tactics. "
                             "Running the same field under two of these says "
                             "whether a party ranking is a fact about the "
                             "parties or about the evaluation")
    parser.add_argument("--preview-iterations", type=int, default=None,
                        help="simulations for the team preview alone, on both "
                             "sides. Off by default so a re-run reproduces an "
                             "earlier one; worth setting when the question is "
                             "which party is best under the agent as deployed")
    parser.add_argument("--checkpoint", default=None,
                        help="a saved network for the prior and leaf value, "
                             "on both sides")
    parser.add_argument("--trust", type=float, default=1.0)
    parser.add_argument("--format", default="singles",
                        choices=("singles", "doubles"))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--reverse", action="store_true",
                        help="walk the schedule from the last fixture back. "
                             "Two machines sharing one --out through git can "
                             "then start at opposite ends and meet in the "
                             "middle, since the seeds come from the fixture "
                             "rather than its position")
    parser.add_argument("--out", default=None,
                        help="write the fixtures and standings here as JSON")
    parser.add_argument("--matrix", action="store_true",
                        help="also print who beat whom")
    args = parser.parse_args()

    parties = ranker_parties(args.parties)
    if args.entrants:
        entrants = tuple(int(one) for one in args.entrants.split(","))
        for one in entrants:
            if not 0 <= one < len(parties):
                parser.error(f"party {one} does not exist; "
                             f"there are {len(parties)}")
    else:
        entrants = tuple(range(len(parties)))
    if len(entrants) < 2:
        parser.error("a tournament needs at least two entrants")

    iterations = args.search_iterations
    config = TournamentConfig(
        battle_format=args.format, checkpoint=args.checkpoint,
        trust=args.trust, parties=args.parties,
        search=SearchConfig(iterations=iterations,
                            preview_iterations=args.preview_iterations,
                            evaluation=args.evaluation,
                            determinizations=max(4, iterations // 20)))
    schedule = fixtures(entrants, args.repeats, args.opponents)
    workers = args.workers if args.workers is not None else default_workers()
    games = len(schedule) * 2
    print(f"{len(entrants)} parties, {len(schedule)} fixtures, {games} games "
          f"| {iterations} sims"
          f"{' + net' if args.checkpoint else ''} | {workers} workers",
          flush=True)
    print(f"each party plays {2 * args.repeats * (len(entrants) - 1)} games",
          flush=True)

    # A 253-party round robin is 31,878 fixtures and days of wall clock, and
    # holding every result in memory until the end means a crash at hour
    # thirty-nine costs thirty-nine hours. Each fixture is appended to a JSONL
    # beside the output as it lands, and a rerun with the same --out skips what
    # is already in it. The seeds come from the fixture, so a resumed run plays
    # exactly the games the interrupted one would have.
    progress = Path(args.out).with_suffix(".jsonl") if args.out else None
    done: dict[tuple[int, int, int], Result] = {}
    if progress and progress.exists():
        for line in progress.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            done[(row["a"], row["b"], row["repeat"])] = Result(
                row["a"], row["b"], row["a_wins"], row["b_wins"], row["draws"])
        print(f"resuming: {len(done)} fixtures already played", flush=True)

    remaining = [one for one in schedule if tuple(one) not in done]
    if args.reverse:
        # Only the order of play changes. ``play_fixture`` seeds from the
        # repeat index, so a fixture played from this end is bit-for-bit the
        # one the other machine would have played, and the two progress files
        # concatenate. Whoever pulls last skips the overlap where they met.
        remaining.reverse()
    results: list[Result] = list(done.values())
    started = beat = time.perf_counter()
    handle = progress.open("a", encoding="utf-8") if progress else None
    try:
        for index, one in enumerate(stream(config, remaining, workers), 1):
            results.append(one)
            if handle:
                # Which repeat this was is not on the Result, and the pair is
                # enough to line it up again: fixtures are generated in order,
                # so the nth sighting of a pair is its nth repeat.
                seen = sum(1 for r in results
                           if (r.a, r.b) == (one.a, one.b)) - 1
                handle.write(json.dumps({
                    "a": one.a, "b": one.b, "repeat": seen,
                    "a_wins": one.a_wins, "b_wins": one.b_wins,
                    "draws": one.draws}) + "\n")
                handle.flush()
            now = time.perf_counter()
            if now - beat >= 30.0 or index == len(remaining):
                rate = index / max(now - started, 1e-9)
                left = (len(remaining) - index) / rate
                print(f"  {len(results)}/{len(schedule)} fixtures  "
                      f"~{left / 3600:.1f}h left", flush=True)
                beat = now
    finally:
        if handle:
            handle.close()

    rows = standings(results, entrants)
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({
            "format": args.format,
            "iterations": iterations,
            "checkpoint": args.checkpoint,
            "repeats": args.repeats,
            "entrants": list(entrants),
            "standings": [
                {"party": row.party, "title": parties[row.party].title,
                 "wins": row.wins, "losses": row.losses, "draws": row.draws,
                 "rate": row.rate}
                for row in rows
            ],
            "fixtures": [
                {"a": one.a, "b": one.b, "a_wins": one.a_wins,
                 "b_wins": one.b_wins, "draws": one.draws}
                for one in results
            ],
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  written to {target}")

    width = max(len(_name(parties[row.party])) for row in rows)
    print(f"\n  {'#':>2}  {'party':>5}  {'win rate':>20}  {'W-L-D':>12}  team")
    for place, row in enumerate(rows, 1):
        low, high = wilson(row.wins, row.decided)[1:]
        span = f"{row.rate:6.1%} [{low:.1%}, {high:.1%}]"
        print(f"  {place:>2}  {row.party:>5}  {span:>20}  "
              f"{row.wins}-{row.losses}-{row.draws:<6}  "
              f"{_name(parties[row.party])[:width]}")

    if args.matrix:
        _print_matrix(results, entrants)

    # Whether the field is separated at all. Twenty teams and a hundred-odd
    # games each will produce a first place whatever happens; the question is
    # whether first place is above the field or merely at the top of it.
    best, worst = rows[0], rows[-1]
    if wilson(best.wins, best.decided)[1] > 0.5:
        print(f"\n  party {best.party} is above the field, separably.")
    else:
        print(f"\n  party {best.party} leads but its interval covers 50% -- "
              f"more repeats, or the field is flat.")
    print(f"  spread: {best.rate:.1%} down to {worst.rate:.1%}")

    return 0


def _name(party) -> str:
    return party.title


def _print_matrix(results, entrants) -> None:
    """Row's win rate against column, so a leader's losses are visible."""
    wins: dict[tuple[int, int], list[int]] = {}
    for one in results:
        got = wins.setdefault((one.a, one.b), [0, 0])
        got[0] += one.a_wins
        got[1] += one.b_wins
    print("\n  head to head (row's wins over column, blank is the diagonal)")
    print("      " + "".join(f"{b:>5}" for b in entrants))
    for a in entrants:
        cells = []
        for b in entrants:
            if a == b:
                cells.append("    .")
                continue
            got = wins.get((a, b))
            if got is None:
                got = list(reversed(wins.get((b, a), [0, 0])))
            total = got[0] + got[1]
            cells.append(f"{got[0] / total:>5.0%}" if total else "    -")
        print(f"  {a:>3} " + "".join(cells))


if __name__ == "__main__":
    raise SystemExit(main())
