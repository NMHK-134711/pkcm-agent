"""Read the book of parties a search left behind.

    python scripts/party_notebook.py runs/notebook_salamence.jsonl
    python scripts/party_notebook.py runs/notebook_salamence.jsonl --show 3
    python scripts/party_notebook.py runs/notebook_salamence.jsonl \
        --export data/champions/parties_evolved.json --top 12

``evolve_party.py`` writes every candidate it grades, survivors and casualties
alike, so the book is most of what a run cost rather than the four teams that
came out of it.

**Two things to keep in mind while reading it.**

Rows are sorted by the *lower* end of the floor's interval, not the floor. A
party knocked out in the first round was graded on a handful of games and was
knocked out partly for being unlucky; sorting by the score would put those
draws on top, which is the winner's curse the round robin already taught us to
distrust. Sorting by the lower bound means a party has to have been measured
before it can rank, and a promising one with forty games behind it is exactly
the thing to re-judge rather than to trust.

And rows are grouped by *basis* -- the field, the subset, the agent. Numbers
from different bases are not comparable and this does not mix them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:                       # pragma: no cover - piping
        pass

from pkcm.data.dex import load_dex                              # noqa: E402
from pkcm.train.party_evolve import Notebook                    # noqa: E402


def show(dex, row: dict) -> None:
    """One party in full, in the shape the .txt parties are read in."""
    for one in row["team"]:
        species = dex.species[one["species"]]
        item = f" @ {dex.items[one['item']].name}" if one.get("item") else ""
        ability = one.get("ability") or "-"
        print(f"    {species.name}{item}")
        print(f"      {ability} | {one['nature']} | "
              f"{'/'.join(str(value) for value in one['sp'])}")
        print(f"      {', '.join(dex.moves[move].name for move in one['moves'])}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="the .jsonl a run wrote")
    parser.add_argument("--top", type=int, default=15,
                        help="rows per basis")
    parser.add_argument("--show", type=int, default=0,
                        help="print this many of them in full")
    parser.add_argument("--min-games", type=int, default=0,
                        help="ignore rows measured on fewer games than this. "
                             "The honest way to read a book that mixes a "
                             "first-round grading with a judged one")
    parser.add_argument("--export", default=None,
                        help="write the top rows as a parties file, so they "
                             "can be run through judge.py or a round robin")
    args = parser.parse_args()

    dex = load_dex()
    notebook = Notebook.open(args.path)
    if not notebook.seen:
        print(f"{args.path}: nothing in it yet")
        return 0

    kept = {key: row for key, row in notebook.seen.items()
            if row["games"] >= args.min_games}
    print(f"{args.path}: {len(notebook.seen)} parties"
          + (f", {len(kept)} with {args.min_games}+ games"
             if args.min_games else ""))

    exported = []
    for basis in sorted({row["basis"] for row in kept.values()}):
        rows = sorted((row for row in kept.values() if row["basis"] == basis),
                      key=lambda one: (-one["low"], -one["floor"]))[:args.top]
        print(f"\n=== {basis} === ({len(rows)} shown)")
        for rank, row in enumerate(rows, 1):
            names = " / ".join(dex.species[one["species"]].name
                               for one in row["team"])
            print(f"  {rank:2}. floor {row['floor']:.3f} "
                  f"[{row['low']:.3f}, {row['high']:.3f}] "
                  f"mean {row['mean']:.3f} worst {row['worst']:.3f} "
                  f"{row['games']:4} games  {row['stage']:10} {row['origin'][:24]}")
            print(f"      {names}")
            if rank <= args.show:
                show(dex, row)
        exported.extend(rows)

    if args.export:
        # A bare list, which is what _party_payload reads: these are meant to
        # be handed straight to judge.py or a round robin as a field.
        payload = [
            {"id": f"nb{index + 1}",
             "title": f"{row['floor']:.3f} floor, {row['games']} games, "
                      f"{row['stage']}",
             "source": "evolve", "team": row["team"]}
            for index, row in enumerate(exported)]
        Path(args.export).write_text(
            json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8")
        print(f"\nwrote {len(exported)} parties to {args.export}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
