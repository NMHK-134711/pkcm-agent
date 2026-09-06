"""One field file out of the hand parties and the imported ones, deduplicated.

    python scripts/build_field.py

The 46 parties this project has measured everything on were entered by hand
from ranker screenshots; the 208 from pokesol were read out of the articles
their authors published. Both are real teams off the same ladder, so the same
team can be in both, and a field that counts one twice weights it twice.

**Identical means identical.** Same six species, and for each of them the same
ability, item, nature, four moves and six stat points. Two parties that differ
by a single point of Defence are two parties: whoever moved that point moved
it for a reason, and this is not the place to decide the reason was wrong.
Move *order* is not part of it -- the same four moves in another order is the
same set -- and that is reported separately so the choice is visible rather
than assumed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pkcm.data.dex import load_dex  # noqa: E402
from pkcm.engine.legality import ranker_parties, team_errors  # noqa: E402


def as_entry(one) -> dict:
    return {"species": one.species, "ability": one.ability,
            "moves": list(one.moves), "item": one.item,
            "nature": one.nature, "sp": list(one.sp)}


def signature(team, ordered_moves: bool = True):
    """What makes two parties the same party."""
    def one(entry):
        moves = tuple(entry["moves"]) if ordered_moves else tuple(sorted(entry["moves"]))
        return (entry["species"], entry["ability"], entry["item"],
                entry["nature"], tuple(entry["sp"]), moves)

    return frozenset(Counter(one(entry) for entry in team).items())


def main() -> int:
    for out in (sys.stdout, sys.stderr):
        try:
            out.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imported", default=str(ROOT / "runs" / "pokesol_parties.json"))
    parser.add_argument("--regulation", default="m_b")
    parser.add_argument("--out", default=str(ROOT / "data" / "champions" / "parties_field.json"))
    args = parser.parse_args()

    dex = load_dex()
    regulation = dex.regulation(args.regulation)

    rows = []
    for index, party in enumerate(ranker_parties()):
        rows.append({"id": f"hand-{index}", "title": party.title,
                     "source": "hand", "team": [as_entry(one) for one in party.team]})
    imported = json.loads(Path(args.imported).read_text(encoding="utf-8"))["parties"]
    for party in imported:
        rows.append({"id": party["source"].replace(".html", ""),
                     "title": f"pokesol rank {party['rank']}",
                     "source": "pokesol", "url": party["url"], "rank": party["rank"],
                     "rating": party["rating"],
                     "team": [{k: v for k, v in one.items() if not k.startswith("_")}
                              for one in party["team"]]})
    print(f"hand {sum(1 for r in rows if r['source'] == 'hand')}, "
          f"imported {sum(1 for r in rows if r['source'] == 'pokesol')}, "
          f"{len(rows)} before deduplication")

    seen, kept, dropped = {}, [], []
    for row in rows:
        key = signature(row["team"])
        if key in seen:
            dropped.append((row, seen[key]))
            continue
        seen[key] = row
        kept.append(row)

    # The same six sets with two moves swapped round is the same party. Counted
    # but not removed, because hk asked for identical.
    loose: dict = {}
    also = 0
    for row in kept:
        key = signature(row["team"], ordered_moves=False)
        if key in loose:
            also += 1
        else:
            loose[key] = row

    print(f"exact duplicates removed: {len(dropped)}")
    for row, first in dropped[:10]:
        print(f"   {row['title'][:34]:36} == {first['title'][:34]}")
    print(f"identical but for the order of some moves, kept: {also}")

    illegal = [row for row in kept
               if team_errors(dex, regulation,
                              tuple(__import__("pkcm.engine.legality", fromlist=["_party_set"])
                                    ._party_set(one) for one in row["team"]))]
    if illegal:
        print(f"WARNING: {len(illegal)} parties in the field do not pass the rules")
        for row in illegal[:5]:
            print(f"   {row['title']}")

    Path(args.out).write_text(
        json.dumps([{k: v for k, v in row.items()} for row in kept],
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{len(kept)} parties -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
