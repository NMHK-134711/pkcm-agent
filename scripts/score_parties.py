"""Score a file of parties against the field and write down *why* the number is
what it is.

    python scripts/score_parties.py --parties data/champions/parties_handmade.json
    python scripts/score_parties.py --parties ... --basis judge

``party_evolve``'s notebook keeps the floor and the bring rates. It does not
keep the one thing a person needs in order to fix a party: **which opponents
are the tail**. The floor is the mean of the worst quarter, so the tail is the
answer -- and it was being computed, folded into a number, and thrown away. A
row here carries the six worst matchups with their species, so the next party
can be built against something rather than around a number.

Two bases, both from ``party_floor``:

    search   40 opponents, 200 sims, no rollout   -- cheap, for comparing
    judge    80 opponents, 400 sims, 20-turn rollout -- the number to quote

They are not comparable with each other and the basis string says so, exactly
as the evolve notebook does. Rows are appended; a party already scored at a
basis is skipped unless it has fewer games than this run would give it.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:                       # pragma: no cover - piping
        pass

from pkcm.engine.legality import ranker_parties          # noqa: E402
from pkcm.engine.pokemon import PokemonSet               # noqa: E402
from pkcm.search import SearchConfig                     # noqa: E402
from pkcm.train.parallel import default_workers          # noqa: E402
from pkcm.train.party_evolve import basis_of, signature  # noqa: E402
from pkcm.train.party_floor import FloorConfig, score    # noqa: E402

FIELD = str(ROOT / "data/champions/parties_field.json")


def a_team(party: dict) -> tuple[PokemonSet, ...]:
    return tuple(PokemonSet(species=one["species"], ability=one["ability"],
                            moves=tuple(one["moves"]), item=one["item"],
                            nature=one["nature"], sp=tuple(one["sp"]))
                 for one in party["team"])


def config_for(basis: str, parties: str, regulation: str) -> FloorConfig:
    if basis == "judge":
        return FloorConfig(parties=parties, regulation=regulation, seed_games=4,
                           opponents=80,
                           search=SearchConfig(iterations=400, determinizations=20,
                                               rollout_turns=20,
                                               rollout_policy="greedy"))
    return FloorConfig(parties=parties, regulation=regulation, seed_games=4,
                       opponents=40,
                       search=SearchConfig(iterations=200, determinizations=10))


def already(path: pathlib.Path) -> dict[tuple[str, str], int]:
    """(basis, signature) -> games, so a re-run does not repeat itself."""
    seen: dict[tuple[str, str], int] = {}
    if not path.exists():
        return seen
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["basis"], row["signature"])
        seen[key] = max(seen.get(key, 0), row["games"])
    return seen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parties", required=True,
                        help="the file of parties to score")
    parser.add_argument("--field", default=FIELD,
                        help="what they are measured against")
    parser.add_argument("--regulation", default="m_c")
    parser.add_argument("--basis", default="search", choices=("search", "judge"))
    parser.add_argument("--budget", type=int, default=400,
                        help="battles per party")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", default=str(ROOT / "runs/notebook_handmade.jsonl"))
    parser.add_argument("--only", default=None,
                        help="comma-separated party ids, for re-scoring one")
    args = parser.parse_args()

    field = ranker_parties(args.field)
    settings = config_for(args.basis, args.field, args.regulation)
    basis = basis_of(type("_", (), {"floor": settings})())
    workers = args.workers if args.workers is not None else default_workers()

    parties = json.loads(pathlib.Path(args.parties).read_text(encoding="utf-8"))
    if args.only:
        wanted = {one.strip() for one in args.only.split(",")}
        parties = [one for one in parties if one["id"] in wanted]

    out = pathlib.Path(args.out)
    done = already(out)
    print(f"{len(parties)} parties, {args.budget} battles each, {workers} workers")
    print(f"basis {basis}")
    if done:
        print(f"notebook {out.name}: {len(done)} rows already there")

    for party in parties:
        team = a_team(party)
        key = (basis, signature(team))
        if done.get(key, 0) >= args.budget:
            print(f"\n{party['id']} already scored on this basis; skipping")
            continue
        started = time.time()
        floor = score(team, settings, budget=args.budget, workers=workers)
        worst = [{"title": field[one.opponent].title,
                  "wins": one.wins, "losses": one.losses,
                  "team": [f"{m.species}@{m.item}" for m in field[one.opponent].team]}
                 for one in floor.worst(6)]
        row = {
            "id": party["id"], "title": party["title"],
            "signature": signature(team), "basis": basis,
            "floor": round(floor.cvar, 4), "low": round(floor.low, 4),
            "high": round(floor.high, 4), "mean": round(floor.mean, 4),
            "worst": round(floor.minimum, 4), "games": floor.games,
            "live": floor.selection.live(),
            "rigidity": round(floor.selection.rigidity, 3),
            "bring": [round(one, 3) for one in floor.selection.rates],
            # hk, on the rarely-brought slot: that is what a complement is.
            # The bring rate alone cannot tell a party's answer to three bad
            # matchups from a passenger, so the win rate *while on the field*
            # is written down beside it, and ``carrying`` names the slots that
            # are brought seldom and win anyway.
            "slot_win": [round(one, 3) for one in floor.selection.win_rates],
            "carrying": list(floor.selection.carrying()),
            "combos": [[list(combo), count, wins]
                       for combo, count, wins in floor.selection.combos[:4]],
            # The tail itself, which is the whole reason this file exists.
            "worst_opponents": worst,
            "minutes": round((time.time() - started) / 60, 1),
            "why": party.get("why", ""),
            "team": party["team"],
        }
        with out.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"\n{party['id']} {party['title']}")
        print(f"  floor {row['floor']} [{row['low']}, {row['high']}]  mean "
              f"{row['mean']}  worst {row['worst']}  {row['games']} games  "
              f"live {row['live']}  rigid {row['rigidity']}  ({row['minutes']}m)")
        names = [one["species"] for one in party["team"]]
        print("  " + "  ".join(
            f"{name}{'*' if index in floor.selection.carrying() else ''} "
            f"{bring:.0%}/{win:.0%}"
            for index, (name, bring, win)
            in enumerate(zip(names, row["bring"], row["slot_win"]))))
        print("  (brought%/won% -- * = brought seldom and wins when it is)")
        for entry in worst:
            print(f"    {entry['wins']}-{entry['losses']}  {entry['title']:20s}"
                  f"  {' / '.join(entry['team'])}")
        sys.stdout.flush()
    print("\ndone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
