"""The articles the importer would not take, as a list to work through.

    python scripts/held_articles.py
    python scripts/held_articles.py --out runs/held.md

An import that guesses is worse than one that stops, so 81 of pokesol's 241
are held. This says which, why, and what is already known about each -- the
link, the rank, the six species the search index gives, and which of them the
cards accounted for -- so filling one in is reading one article rather than
starting from nothing.

Sorted by how little is left to do: the ones missing a single Pokemon first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    for out in (sys.stdout, sys.stderr):
        try:
            out.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parties", default=str(ROOT / "runs" / "pokesol_parties.json"))
    parser.add_argument("--index", default=str(ROOT / "data" / "champions" / "articles.json"))
    parser.add_argument("--adjudicated",
                        default=str(ROOT / "data" / "champions" / "held_adjudicated.json"),
                        help="articles hk has already ruled on; they come out "
                             "of the list so what is left is only work")
    parser.add_argument("--out", default=str(ROOT / "runs" / "held_articles.md"))
    args = parser.parse_args()

    data = json.loads(Path(args.parties).read_text(encoding="utf-8"))
    index = {a["url"]: a for a in
             json.loads(Path(args.index).read_text(encoding="utf-8"))["articles"]}

    ruled = {}
    settled = Path(args.adjudicated)
    if settled.exists():
        ruled = {k: v for k, v in
                 json.loads(settled.read_text(encoding="utf-8")).items()
                 if not k.startswith("_")}

    rows = []
    for record in data["held"]:
        if record["url"] in ruled:
            continue
        row = index.get(record["url"] or "", {})
        have = {one["species"] for one in record["team"]}
        rows.append({
            "url": record["url"],
            "rank": record["rank"], "rating": record["rating"],
            "left": 6 - len(record["team"]),
            "have": sorted(have),
            "six": [t["name"] for t in row.get("team", [])],
            "items": [i["name"] for i in row.get("items", [])],
            "why": record["problems"][0] if record["problems"] else "",
        })
    rows.sort(key=lambda r: (r["left"], r["rank"] or 99999))

    lines = ["# Held pokesol articles",
             "",
             f"{len(rows)} still to look at, of 241 read. The importer takes a "
             "party only when every one of the six is settled by the article "
             "itself, so these are the ones where something is missing rather "
             "than wrong.",
             ""]
    if ruled:
        lines += [f"{len(ruled)} more have been ruled on already and are not "
                  "listed; they are in data/champions/held_adjudicated.json.",
                  ""]
    for row in rows:
        lines.append(f"## {row['left']} left — rank {row['rank']} "
                     f"(rating {row['rating']})")
        lines.append("")
        lines.append(f"- {row['url']}")
        lines.append(f"- the six: {', '.join(row['six'])}")
        lines.append(f"- items: {', '.join(row['items'])}")
        if row["have"]:
            lines.append(f"- already read: {', '.join(row['have'])}")
        lines.append(f"- why: {row['why']}")
        lines.append("")

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    short = sum(1 for r in rows if r["left"] == 1)
    empty = sum(1 for r in rows if r["left"] == 6)
    print(f"{len(rows)} held: {short} missing one Pokemon, {empty} with nothing "
          f"read at all, {len(rows) - short - empty} in between")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
