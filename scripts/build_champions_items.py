"""Turn the scraped op.gg item list into our Champions item roster.

``포챔스 아이템 목록.txt`` is the item list as it stands in the live game,
scraped from op.gg. It is more current than anything else we have: Showdown's
Champions mod is a community port and can lag, while this is the shop screen.

The file is Korean, so the names are matched against PokeAPI's multilingual item
names (one CSV, all languages) to recover Showdown ids. Anything that fails to
match is reported rather than dropped -- an unmatched name is a bug in this
script or a genuinely new item, and both need a person to look.

Output: ``data/champions/items_m_b.json`` (committed).

Usage:
    python scripts/build_champions_items.py
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "포챔스 아이템 목록.txt"
OUT_PATH = ROOT / "data" / "champions" / "items_m_b.json"

POKEAPI_CSV = "https://raw.githubusercontent.com/PokeAPI/pokeapi/master/data/v2/csv/"
KOREAN_LANGUAGE_ID = "3"
ENGLISH_LANGUAGE_ID = "9"

#: Names op.gg writes differently from PokeAPI, or that PokeAPI does not carry.
MANUAL_IDS = {
    "기합의머리띠": "focusband",
    "기합의띠": "focussash",
    "생명의구슬": "lifeorb",
    "먹다남은음식": "leftovers",
    "구애스카프": "choicescarf",
    "구애머리띠": "choiceband",
    "구애안경": "choicespecs",
    "돌격조끼": "assaultvest",
    "진화의휘석": "eviolite",
    "안전고글": "safetygoggles",
    "두꺼운장갑": "punchingglove",
    "부적동전": "amuletcoin",
    # Meowstic's Mega is one option in game but two formes in Showdown
    # (meowsticmmega / meowsticfmega), so the species-side lookup misses it.
    "냐오닉스나이트": "meowsticite",
    # The game drops the final consonant: 후딘 -> 후디나이트, not 후딘나이트.
    "후디나이트": "alakazite",
}


def to_id(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def fetch_csv(name: str) -> list[dict[str, str]]:
    request = urllib.request.Request(POKEAPI_CSV + name,
                                     headers={"User-Agent": "pkcm-agent/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        text = response.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def korean_mega_stones() -> dict[str, str]:
    """``<species>나이트`` -> the Mega Stone id, for Champions' new stones.

    PokeAPI has never heard of Chandelurite or Meganiumite, but Showdown has,
    and the Korean name is just the species plus 나이트 (plus X or Y where the
    species has two Megas). So go species-side instead of item-side.
    """
    from pkcm.data.dex import load_dex

    dex = load_dex()
    identifiers = {row["id"]: row["identifier"] for row in fetch_csv("pokemon_species.csv")}
    korean_species: dict[str, str] = {}
    for row in fetch_csv("pokemon_species_names.csv"):
        if row["local_language_id"] != KOREAN_LANGUAGE_ID:
            continue
        identifier = identifiers.get(row["pokemon_species_id"])
        if identifier:
            korean_species[row["name"].replace(" ", "")] = to_id(identifier)

    # Mega species id -> the stone that produces it.
    stone_for_mega: dict[str, str] = {}
    for item_id, mega_ids in dex.mega_stones.items():
        for mega_id in mega_ids:
            stone_for_mega[mega_id] = item_id

    mapping: dict[str, str] = {}
    for korean, species_id in korean_species.items():
        for suffix, forme in (("", "mega"), ("X", "megax"), ("Y", "megay")):
            stone = stone_for_mega.get(species_id + forme)
            if stone:
                mapping[f"{korean}나이트{suffix}"] = stone
    return mapping


def korean_to_showdown() -> dict[str, str]:
    """Korean item name -> Showdown item id, via PokeAPI's identifiers."""
    identifiers = {row["id"]: row["identifier"] for row in fetch_csv("items.csv")}
    names = fetch_csv("item_names.csv")

    mapping: dict[str, str] = {}
    for row in names:
        if row["local_language_id"] != KOREAN_LANGUAGE_ID:
            continue
        identifier = identifiers.get(row["item_id"])
        if identifier:
            mapping[row["name"].replace(" ", "")] = to_id(identifier)
    return mapping


PRICE_RE = re.compile(r"^(-|[\d,]+)$")


def parse_scrape(text: str) -> list[dict]:
    """Blocks of: name, name again, optional NEW, description, source, price.

    The source is *not* matched against a fixed list. An earlier version
    accepted only Shop/Beginning/Event and silently swallowed the eleven items
    whose source is "Mega Evolution Tutorial" or "Deposit <x> from Pokemon
    Legends: Z-A" -- the block boundary was missed and two entries merged into
    one. The last line before a price is the source, whatever it says.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    entries: list[dict] = []
    index = 0
    while index < len(lines):
        name = lines[index]
        # The scrape repeats the name; that repetition is how a block starts.
        if index + 1 >= len(lines) or lines[index + 1] != name:
            index += 1
            continue
        index += 2

        is_new = index < len(lines) and lines[index] == "NEW"
        if is_new:
            index += 1

        body = []
        while index + 1 < len(lines) and not PRICE_RE.match(lines[index + 1]):
            body.append(lines[index])
            index += 1

        source = lines[index] if index < len(lines) else "?"
        index += 1
        price = lines[index] if index < len(lines) else "-"
        index += 1
        description_parts = body

        entries.append({
            "korean": name,
            "new_in_champions": is_new,
            "description_ko": " ".join(description_parts),
            "source": source,
            "price": price,
        })
    return entries


def main() -> int:
    if not SOURCE.exists():
        print(f"missing {SOURCE}", file=sys.stderr)
        return 1

    entries = parse_scrape(SOURCE.read_text(encoding="utf-8"))
    mapping = korean_to_showdown()
    mapping.update(korean_mega_stones())
    mapping.update(MANUAL_IDS)

    unmatched = []
    for entry in entries:
        key = entry["korean"].replace(" ", "")
        item_id = mapping.get(key)
        entry["id"] = item_id
        if item_id is None:
            unmatched.append(entry["korean"])

    # op.gg's scrape omits 12 Mega Stones whose Megas are M-B legal, so it is
    # not complete on the stone side. The legal Mega list settles those: if the
    # forme is legal, the stone that reaches it must exist.
    from pkcm.data.dex import load_dex

    dex = load_dex()
    regulation = dex.regulation("m_b")
    scraped = {entry["id"] for entry in entries}
    inferred = sorted(
        item_id for item_id, megas in dex.mega_stones.items()
        if item_id not in scraped and any(m in regulation.legal_megas for m in megas)
    )

    payload = {
        "source": "op.gg Pokemon Champions item list, scraped by hk",
        "inferred_mega_stones": inferred,
        "inferred_note": (
            "Not in the op.gg scrape, but their Mega formes are legal in "
            "Regulation M-B, so the stones exist. Showdown's Champions mod "
            "enables all 12 as well."
        ),
        "note": (
            "The item roster as it stands in the live game. More current than "
            "Showdown's Champions mod, which is a community port. Korean names "
            "were matched to Showdown ids through PokeAPI's multilingual names."
        ),
        "count": len(entries),
        "items": entries,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"  mega stones inferred from the legal Mega list: {len(inferred)}")
    matched = len(entries) - len(unmatched)
    print(f"parsed {len(entries)} items, matched {matched}, unmatched {len(unmatched)}")
    print(f"  new in Champions: {sum(1 for e in entries if e['new_in_champions'])}")
    if unmatched:
        print("\nUNMATCHED -- add to MANUAL_IDS:")
        for name in unmatched:
            print(f"  {name}")
    print(f"\nwritten -> {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
