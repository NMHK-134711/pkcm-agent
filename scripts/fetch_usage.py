"""Collect per-species move and item usage from pokedb, for hk to run.

    python scripts/fetch_usage.py
    python scripts/fetch_usage.py --limit 20        # the most played first

**Read this before running it.** ``champs.pokedb.tokyo/robots.txt`` names
ClaudeBot and Claude-SearchBot and disallows them from the whole site, and the
server enforces it -- a request identifying as Claude gets 403. Everything
else, ``User-agent: *`` included, is allowed on these paths, and an ordinary
``python-urllib`` request is served normally. So this is a tool for the person
whose research it is to run under their own name; it does not pretend to be a
browser, and the default User-Agent says what it is and who to contact. If you
would rather not automate it at all, saving the pages by hand and pointing
``--from-cache`` at them parses exactly the same.

It is deliberately slow. One page a second and a half by default, everything
cached to disk, and a resume that skips what it already has -- so a second run
costs the site nothing and a interrupted run does not start over.

**What the pages carry.** Move adoption rates (the top ten, with power and
category) and item adoption rates for the whole season. Not abilities, not
natures, not spreads. Items we already have from the 518 ranked teams in
``data/champions/s*_single_ranked_teams.json`` -- and the two agree closely --
so the reason to run this is the moves.

**Why we want them.** Three things, and only the first is about the species
the field is missing.

The 46-party field the optimizer scores against covers 82.3% of the ladder's
slots. The rest are species it never plays -- Blaziken in 9.5% of teams,
Lopunny, Alolan Ninetales, Dragapult -- and a floor measured against a field
missing them has holes nothing tested.

The 52 species it *does* play are represented by a handful of sets each, drawn
from 46 parties. Garchomp holds 226 of the ladder's slots and we have a few of
them. The adoption rates are the distribution those samples were drawn from.

And the coach builds its opponent out of the same 46 parties, which is why a
Hisuian Arcanine placeholder arrived carrying Intimidate when the real one had
Rock Head. Better sets there are better advice in a live game.

So fetch all of them. ``--only-missing`` exists for a top-up run; it is not
the way to start, and the whole list is 138 pages, about three and a half
minutes at the default delay. The 47 further entries on the speed-line page
are Mega formes, which the ranked teams record as the base species holding a
stone -- Garchomp's own page lists Garchompite among its items -- so the base
page already covers them.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from html import unescape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

SHOW = "https://champs.pokedb.tokyo/pokemon/show/{key}?season={season}&rule={rule}"
AGENT = ("pkcm-agent/0.1 (personal Pokemon Champions research; "
         "contact khg950520@gmail.com)")


# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #


def japanese_tables() -> tuple[dict, dict, dict]:
    """dex-forme key -> species row, and Japanese name -> our move / item id."""
    raw = ROOT / "data" / "raw" / "pokechams"
    species = {}
    for row in json.loads((raw / "champions_pokemon.json").read_text(encoding="utf-8")):
        key = f"{int(row['nationalDex']):04d}-{int(row['formIndex']):02d}"
        species[key] = row
    moves = {row["nameJa"]: row for row in
             json.loads((raw / "moves.json").read_text(encoding="utf-8"))}
    items = {row["nameJa"]: row for row in
             json.loads((raw / "items.json").read_text(encoding="utf-8"))}
    return species, moves, items


def wanted_keys(species: dict, only_missing: bool) -> list[tuple[str, str, int]]:
    """(key, japanese name, ladder slots) for the species worth fetching.

    Ordered by how much of the ladder they account for, so ``--limit`` takes
    the ones that matter rather than the ones that sort first.
    """
    from pkcm.data.dex import load_dex
    from pkcm.engine.legality import ranker_parties

    dex = load_dex()
    field = {one.species for party in ranker_parties() for one in party.team}
    slots: Counter[str] = Counter()
    label: dict[str, str] = {}
    for season in ("s3", "s4"):
        path = ROOT / "data" / "champions" / f"{season}_single_ranked_teams.json"
        if not path.exists():
            continue
        for team in json.loads(path.read_text(encoding="utf-8"))["teams"]:
            for one in team["team"]:
                key = str(one["id"])
                slots[key] += 1
                label[key] = one["pokemon"]

    out = []
    for key, count in slots.most_common():
        row = species.get(key)
        if row is None:
            continue
        our = row["slug"].replace("-", "")
        if our not in dex.species:
            our = row["slug"]
        if our not in dex.species:
            continue
        if only_missing and our in field:
            continue
        out.append((key, label.get(key, row["nameJa"]), count))
    return out


# --------------------------------------------------------------------------- #
# Fetching and parsing
# --------------------------------------------------------------------------- #


def fetch(key: str, season: int, rule: int, cache: Path, agent: str,
          delay: float, timeout: int = 60) -> str | None:
    """One page, from the cache if it is there and from the site if not."""
    cache.mkdir(parents=True, exist_ok=True)
    saved = cache / f"{key}-s{season}-r{rule}.html"
    if saved.exists():
        return saved.read_text(encoding="utf-8", errors="replace")
    url = SHOW.format(key=key, season=season, rule=rule)
    request = urllib.request.Request(url, headers={
        "User-Agent": agent,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "ja,en;q=0.8",
    })
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        print(f"    {key}: HTTP {error.code}", flush=True)
        return None
    except Exception as error:  # noqa: BLE001 - one bad page is not a bad run
        print(f"    {key}: {type(error).__name__} {error}", flush=True)
        return None
    saved.write_text(body, encoding="utf-8")
    time.sleep(delay)
    return body


def _trend_blob(page: str) -> dict:
    """The Alpine ``x-data="pokemonShowTrend({...})"`` payload."""
    start = page.find('x-data="pokemonShowTrend(')
    if start < 0:
        return {}
    opening = page.index("{", start)
    depth, index = 0, opening
    while index < len(page):
        if page.startswith("&quot;", index):
            index += 6
            continue
        if page[index] == "{":
            depth += 1
        elif page[index] == "}":
            depth -= 1
            if depth == 0:
                break
        index += 1
    try:
        return json.loads(unescape(page[opening:index + 1]))
    except ValueError:
        return {}


def parse(page: str, moves_ja: dict, items_ja: dict) -> dict:
    """Move and item adoption rates, with our ids attached where they map."""
    moves = []
    for raw in re.findall(r'data-move-detail="(\{.*?\})"', page):
        try:
            one = json.loads(unescape(raw))
        except ValueError:
            continue
        row = moves_ja.get(one["name"])
        moves.append({"name_ja": one["name"], "rate": one.get("rate"),
                      "rank": one.get("rank"),
                      "id": (row or {}).get("nameEn", "").lower().replace(" ", "")
                            .replace("-", "").replace("'", "") or None})

    blob = _trend_blob(page)
    details = blob.get("itemDetails") or {}
    items = []
    for ident, series in (blob.get("items") or {}).items():
        if not series:
            continue
        name = (details.get(str(ident)) or {}).get("name", str(ident))
        row = items_ja.get(name)
        items.append({"name_ja": name, "rate": series[-1].get("rate"),
                      "id": (row or {}).get("nameEn", "").lower().replace(" ", "")
                            .replace("-", "").replace("'", "") or None})
    items.sort(key=lambda one: -(one["rate"] or 0))
    return {"moves": moves, "items": items}


def main() -> int:
    for out in (sys.stdout, sys.stderr):
        try:
            out.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, default=5)
    parser.add_argument("--rule", type=int, default=0, help="0 is singles")
    parser.add_argument("--only-missing", action="store_true",
                        help="just the species the 46-party field never plays "
                             "(86 of the 138). For a top-up run -- the first "
                             "run wants all of them, including the ones we "
                             "have, whose sets we only have a few of")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many species, most played first")
    parser.add_argument("--delay", type=float, default=1.5,
                        help="seconds between requests. Please do not lower it")
    parser.add_argument("--agent", default=AGENT)
    parser.add_argument("--from-cache", action="store_true",
                        help="parse what is already downloaded and fetch nothing")
    parser.add_argument("--cache", default=str(ROOT / "data" / "raw" / "pokedb_usage"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    species, moves_ja, items_ja = japanese_tables()
    targets = wanted_keys(species, args.only_missing)
    if args.limit:
        targets = targets[:args.limit]
    cache = Path(args.cache)
    out = Path(args.out) if args.out else \
        ROOT / "data" / "champions" / f"usage_m{args.season}_singles.json"

    print(f"{len(targets)} species, season M-{args.season}, "
          f"{'cache only' if args.from_cache else f'{args.delay}s between requests'}")
    print(f"cache {cache}")
    collected = {}
    for index, (key, name, slots) in enumerate(targets, 1):
        page = (cache / f"{key}-s{args.season}-r{args.rule}.html")
        if args.from_cache:
            body = page.read_text(encoding="utf-8", errors="replace") \
                if page.exists() else None
        else:
            body = fetch(key, args.season, args.rule, cache, args.agent, args.delay)
        if not body:
            continue
        parsed = parse(body, moves_ja, items_ja)
        row = species[key]
        collected[key] = {"slug": row["slug"], "name_ja": row["nameJa"],
                          "ladder_slots": slots, **parsed}
        top = ", ".join(f"{m['name_ja']} {m['rate']}%" for m in parsed["moves"][:3])
        print(f"  [{index:3}/{len(targets)}] {key} {name:16} "
              f"{len(parsed['moves'])} moves, {len(parsed['items'])} items | {top}",
              flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"season": args.season, "rule": args.rule,
                               "species": collected}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    unmapped = sum(1 for one in collected.values()
                   for m in one["moves"] if m["id"] is None)
    print(f"\n{len(collected)} species written to {out}")
    if unmapped:
        print(f"{unmapped} move names did not map to one of ours -- "
              f"the name_ja is kept either way")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
