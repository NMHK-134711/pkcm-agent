"""Download the pkmnchamps.com ranker-party archive and its Korean name tables.

https://pkmnchamps.com/archive is a Korean Champions site whose *랭커 파티*
section collects top-ladder teams from X, YouTube and Pokepaste. Every entry is
a complete, ladder-proven Regulation M-B party: six slots with ability, item,
nature, SP spread and four moves. That is the thing this repo has no supervised
source for -- ``docs/DESIGN.md`` lists party construction as the one learning
target with no data behind it.

Two fetches, because the site keeps the two halves in different places:

* **The parties.** ``/archive`` is a Next.js App Router page. Nothing useful is
  in the HTML body -- but the server-component payload *is*, inline, in the
  ``self.__next_f.push([1,"..."])`` script tags. One GET yields every archived
  party with its ``showdown_slots`` (Showdown-style English slugs plus EVs).
  There is no JSON API for the list; ``/api/archive`` is a soft 404 that renders
  the app shell. Individual parties do have one (``/api/parties/<id>``), used
  here only to repair a party the inline payload came up short on.

* **The Korean names.** The site ships its whole dex -- species, moves, items,
  abilities, natures, mega stones -- inside one static JS chunk. Which chunk is
  a build hash, so it is found by scanning the page's chunk list for the table
  markers rather than hardcoded; a redeploy renames the file and this still
  finds it.

Why bother with the site's names when ``data/champions/names.json`` already has
Korean ones: they disagree on exactly the forms that matter. names.json calls
``floette-mega`` 메가플라엣테; the site calls it 메가 플라엣테(영원의 꽃), and
the party samples are meant to read the way the site does.

Output goes to ``data/raw/pkmnchamps/`` (gitignored), with a MANIFEST recording
url, size and sha256 so a stale cache is detectable.

Usage:
    python scripts/fetch_pkmnchamps.py [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://pkmnchamps.com"
ARCHIVE_URL = f"{BASE}/archive"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "pkmnchamps"

#: A plain urllib User-Agent gets 403 at the edge, and the JSON API also wants a
#: same-origin Referer. The browser string is here because the site serves only
#: browsers, not to disguise what this is -- one snapshot, a handful of GETs.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko,en;q=0.8",
    "Referer": f"{BASE}/",
}

#: Politeness delay between requests. A full run costs three or four of them.
DELAY_S = 1.0

#: Inline server-component payload: ``self.__next_f.push([1,"<escaped chunk>"])``.
#: The pieces concatenate into one flight stream; the party array is a JSON
#: substring of it.
FLIGHT_RE = re.compile(r'self\.__next_f\.push\(\[1,(".*?")\]\)</script>', re.S)

#: Every archived party object starts with its uuid. Brace matching is left to
#: ``raw_decode`` -- the flight stream is not JSON as a whole, only in patches.
PARTY_START_RE = re.compile(r'\{"id":"[0-9a-f]{8}-[0-9a-f]{4}-')

CHUNK_RE = re.compile(r"/_next/static/chunks/[A-Za-z0-9%._/-]+\.js")

#: The two chunks we want, identified by what is in them rather than by build
#: hash. The dex is loaded by every page; the mega-stone map only by the party
#: detail page, which is why one archived party gets opened as well.
DEX_MARKERS = ('"damageClassKo"', '"nameKo"')
STONE_MARKERS = ('-mega":"',)


def get(url: str) -> bytes:
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def content_length(url: str) -> int:
    """Size of a static asset without downloading it, 0 if the server won't say."""
    request = urllib.request.Request(url, headers=HEADERS, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return int(response.headers.get("Content-Length") or 0)
    except (urllib.error.URLError, ValueError):
        return 0


# ---------------------------------------------------------------- the parties


def parse_flight(html: str) -> str:
    """Concatenate the inline server-component payload out of the page."""
    pieces = FLIGHT_RE.findall(html)
    if not pieces:
        raise LookupError("no self.__next_f.push payload -- page shape changed")
    return "".join(json.loads(piece) for piece in pieces)


def extract_parties(flight: str) -> list[dict]:
    """Pull every party object out of the flight stream."""
    decoder = json.JSONDecoder()
    found: dict[str, dict] = {}
    for match in PARTY_START_RE.finditer(flight):
        try:
            obj, _ = decoder.raw_decode(flight, match.start())
        except ValueError:
            continue  # a uuid that is not the head of a complete object
        if isinstance(obj, dict) and "showdown_slots" in obj:
            found[obj["id"]] = obj
    return list(found.values())


def slot_count(party: dict) -> int:
    return sum(1 for slot in (party.get("showdown_slots") or []) if slot)


def repair(party: dict) -> dict:
    """Re-fetch one party from the JSON API when the inline copy came up short.

    The flight stream arrives in chunks and a party can straddle a boundary. A
    truncation is silent -- a party with five slots still looks like a party --
    so anything under six slots is checked against the API rather than trusted.
    """
    payload = get(f"{BASE}/api/parties/{party['id']}")
    time.sleep(DELAY_S)
    return json.loads(payload)


# ------------------------------------------------------------- the name table

#: Moves, items and abilities are keyed by slug: ``"liquidation":{"nameKo":...``
KEYED_RE = re.compile(r'"([a-z0-9\'.:%-]+)":(\{"nameKo":")')

#: Alternate formes and natures both lead with ``name``.
FORM_RE = re.compile(r'\{"name":"([a-z0-9-]+)","nameKo":"')

#: Base species use a different record shape, where ``nameEn`` is the slug.
SPECIES_RE = re.compile(r'\{"id":\d+,"nameKo":"[^"]*","nameEn":"([a-z0-9-]+)"')

#: The mega-form -> stone map. Worth having because the archived parties spell
#: the stones half a dozen ways (``starminite`` for ``starmienite``, three
#: spellings of Dragonite's, a bare species slug) and this settles which is
#: meant: a mega slot holds its own stone, whatever the export called it.
MEGA_STONE_RE = re.compile(r'"([a-z0-9-]+-mega(?:-[xy])?)":"([a-z0-9-]+ite)"')


def js_objects(source: str, pattern: re.Pattern[str], group: int) -> dict:
    """Decode every JSON object literal the pattern points at."""
    decoder = json.JSONDecoder()
    out: dict[str, dict] = {}
    for match in pattern.finditer(source):
        try:
            obj, _ = decoder.raw_decode(source, match.start(group))
        except ValueError:
            continue
        out.setdefault(match.group(1), obj)
    return out


def extract_names(chunk: str, stone_chunk: str) -> dict:
    """Split the chunk's one pile of ``nameKo`` records into named tables.

    The chunk is JavaScript, so its string literals carry escapes ``json``
    rejects -- ``\\'`` and ``\\xe9`` (as in "Pok\\xe9mon", which is in a great
    many move descriptions). Either one aborts the record it appears in, and
    silently: the table just comes back short a few hundred moves. Those two are
    normalised away rather than writing a JS parser.
    """
    source = chunk.replace("\\'", "'")
    source = re.sub(r"\\x([0-9a-fA-F]{2})", r"\\u00\1", source)

    keyed = js_objects(source, KEYED_RE, 2)
    moves = {k: v for k, v in keyed.items() if "damageClass" in v}
    items = {k: v for k, v in keyed.items()
             if "category" in v and "damageClass" not in v}
    abilities = {k: v for k, v in keyed.items()
                 if k not in moves and k not in items and "nameEn" in v}

    forms = js_objects(source, FORM_RE, 0)
    natures = {k: v for k, v in forms.items() if "up" in v and "down" in v}
    species = {k: v for k, v in forms.items() if k not in natures}
    for slug, record in js_objects(source, SPECIES_RE, 0).items():
        species.setdefault(slug, record)

    tables = {
        "species": {k: v["nameKo"] for k, v in species.items()},
        "moves": {k: v["nameKo"] for k, v in moves.items()},
        "items": {k: v["nameKo"] for k, v in items.items()},
        "abilities": {k: v["nameKo"] for k, v in abilities.items()},
        "natures": {k: {"ko": v["nameKo"], "up": v["up"], "down": v["down"]}
                    for k, v in natures.items()},
        # Kept because a mega's ability is fixed by its forme and the exported
        # slot still carries the base species' one: Mega Starmie is 천하장사,
        # the party data says natural-cure.
        "species_abilities": {
            k: [a["name"] for a in v.get("abilities") or []]
            for k, v in species.items() if v.get("abilities")
        },
        "mega_stones": dict(MEGA_STONE_RE.findall(stone_chunk.replace("\\'", "'"))),
    }
    thin = [name for name, table in tables.items() if len(table) < 20]
    if thin:
        raise LookupError(f"name tables came out empty: {thin}")
    return tables


def find_chunk(paths: list[str], markers: tuple[str, ...],
               cache: dict[str, str]) -> tuple[str, str]:
    """Return (path, body) of the chunk carrying every marker.

    Ordered biggest-first, which in practice puts the dex chunk on the first
    download; the marker check is what actually decides. Bodies are cached
    because the two chunks are picked out of overlapping candidate lists.
    """
    if not paths:
        raise LookupError("no chunk URLs in the page")
    def cost(path: str) -> tuple[int, int]:
        return (0, 0) if path in cache else (1, -content_length(BASE + path))

    for path in sorted(paths, key=cost):
        if path not in cache:
            cache[path] = get(BASE + path).decode("utf-8", "replace")
            time.sleep(DELAY_S)
        if all(marker in cache[path] for marker in markers):
            return path, cache[path]
    raise LookupError(f"no chunk carrying {markers} among {len(paths)}")


# ----------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the cache is already here")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    parties_path = RAW_DIR / "archive_parties.json"
    names_path = RAW_DIR / "names_ko.json"

    if parties_path.exists() and names_path.exists() and not args.force:
        parties = json.loads(parties_path.read_text(encoding="utf-8"))
        names = json.loads(names_path.read_text(encoding="utf-8"))
        print(f"  archive_parties.json  cached ({len(parties)} parties)")
        print(f"  names_ko.json         cached ({len(names['species'])} species)")
        return 0

    print(f"GET {ARCHIVE_URL}")
    html = get(ARCHIVE_URL).decode("utf-8")
    time.sleep(DELAY_S)

    parties = extract_parties(parse_flight(html))
    if not parties:
        print("no parties in the flight payload -- page shape changed",
              file=sys.stderr)
        return 1

    for index, party in enumerate(parties):
        if slot_count(party) == 6:
            continue
        try:
            fixed = repair(party)
        except (urllib.error.URLError, ValueError) as exc:
            print(f"  ! {party['id']} has {slot_count(party)}/6 slots and the "
                  f"API would not say why: {exc}", file=sys.stderr)
            continue
        if slot_count(fixed) > slot_count(party):
            parties[index] = fixed

    parties.sort(key=lambda p: (p.get("battle_format") or "",
                                -(p.get("rate") or 0), p["id"]))
    complete = sum(1 for p in parties if slot_count(p) == 6)
    print(f"  archive_parties.json  {len(parties):>4} parties "
          f"({complete} with all six slots)")

    # The mega-stone map ships only with the party detail route, so one
    # archived party gets opened for its chunk list. Its body is thrown away.
    detail_html = get(f"{BASE}/parties/{parties[0]['id']}").decode("utf-8")
    time.sleep(DELAY_S)

    cache: dict[str, str] = {}
    archive_chunks = list(dict.fromkeys(CHUNK_RE.findall(html)))
    detail_chunks = list(dict.fromkeys(CHUNK_RE.findall(detail_html)))
    dex_path, dex = find_chunk(archive_chunks, DEX_MARKERS, cache)
    stone_path, stones = find_chunk(detail_chunks, STONE_MARKERS, cache)

    names = extract_names(dex, stones)
    print(f"  names_ko.json         from {dex_path}")
    if stone_path != dex_path:
        print(f"                        and  {stone_path}")
    for table, entries in names.items():
        print(f"      {table:14} {len(entries):>5}")

    manifest: dict[str, dict | str] = {}
    for path, payload, url in (
        (parties_path, parties, ARCHIVE_URL),
        (names_path, names, BASE + dex_path),
    ):
        text = json.dumps(payload, ensure_ascii=False, indent=1)
        path.write_text(text, encoding="utf-8")
        manifest[path.name] = {
            "url": url,
            "bytes": len(text.encode("utf-8")),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
    manifest["_fetched"] = datetime.now(timezone.utc).isoformat()
    (RAW_DIR / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nwrote {RAW_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
