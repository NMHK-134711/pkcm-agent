"""Build the Korean (and English) display-name tables.

Ids stay English everywhere -- they are the keys that tie us to Showdown's data,
the Champions override layer and the scraped item list. Only presentation is
localized, and it happens in the renderer (docs/DESIGN.md §1e).

Names come from PokeAPI's multilingual CSVs. Two things they cannot supply:

* **Formes.** ``pokemon_species_names`` has Gengar but not Gengar-Mega. Korean
  builds those by decoration -- 메가팬텀, 알로라 라이츄 -- so they are composed
  from the base name plus a prefix per forme.
* **Champions originals.** Mega Meganium, Mega Sol, Dragonize and friends
  postdate PokeAPI entirely. Those fall back to English and are reported, so
  the gap is visible rather than silent.

Output: ``data/champions/names.json`` (committed).

Usage:
    python scripts/build_names.py
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
sys.path.insert(0, str(ROOT / "src"))

OUT_PATH = ROOT / "data" / "champions" / "names.json"
POKEAPI_CSV = "https://raw.githubusercontent.com/PokeAPI/pokeapi/master/data/v2/csv/"
KOREAN, ENGLISH = "3", "9"

#: How Korean decorates a forme. Order matters -- the longest suffix wins.
FORME_PREFIXES = (
    ("Mega-X", "메가", " X"),
    ("Mega-Y", "메가", " Y"),
    ("Mega", "메가", ""),
    ("Alola", "알로라 ", ""),
    ("Galar", "가라르 ", ""),
    ("Hisui", "히스이 ", ""),
    ("Paldea-Combat", "팔데아 ", "(컴뱃)"),
    ("Paldea-Blaze", "팔데아 ", "(블레이즈)"),
    ("Paldea-Aqua", "팔데아 ", "(워터)"),
    ("Paldea", "팔데아 ", ""),
)

#: Formes whose Korean name is not a decoration of the base name.
FORME_LITERALS = {
    "Small": "(소사이즈)",
    "Large": "(대사이즈)",
    "Super": "(특대사이즈)",
    "F": "♀",
    "Busted": "(들킨 모습)",
    "Midnight": "(한밤중의 모습)",
    "Dusk": "(황혼의 모습)",
    "Hero": "(마이티폼)",
    "Heat": "(히트로토무)",
    "Wash": "(워시로토무)",
    "Frost": "(프로스트로토무)",
    "Fan": "(스핀로토무)",
    "Mow": "(커트로토무)",
}


def to_id(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def fetch_csv(name: str) -> list[dict[str, str]]:
    request = urllib.request.Request(POKEAPI_CSV + name, headers={"User-Agent": "pkcm-agent/0.1"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return list(csv.DictReader(io.StringIO(response.read().decode("utf-8"))))


def names_by_identifier(table: str, names_table: str, key: str) -> dict[str, str]:
    identifiers = {row["id"]: row["identifier"] for row in fetch_csv(table)}
    mapping: dict[str, str] = {}
    for row in fetch_csv(names_table):
        if row["local_language_id"] != KOREAN:
            continue
        identifier = identifiers.get(row[key])
        if identifier:
            mapping[to_id(identifier)] = row["name"]
    return mapping


def korean_species(base_names: dict[str, str], dex) -> tuple[dict[str, str], list[str]]:
    """Every forme in the dex, decorated from its base species' Korean name."""
    resolved: dict[str, str] = {}
    missing: list[str] = []

    for species in dex.species.values():
        base = base_names.get(species.base_species)
        if base is None:
            missing.append(species.id)
            continue
        if not species.forme:
            resolved[species.id] = base
            continue

        forme = species.forme
        for marker, prefix, suffix in FORME_PREFIXES:
            if forme == marker or forme.startswith(marker + "-"):
                resolved[species.id] = f"{prefix}{base}{suffix}"
                break
        else:
            literal = FORME_LITERALS.get(forme)
            resolved[species.id] = f"{base}{literal}" if literal else f"{base}({forme})"
    return resolved, missing



POKECHAMS_DIR = ROOT / "data" / "raw" / "pokechams"
PKMNCHAMPS_DIR = ROOT / "data" / "raw" / "pkmnchamps"


def korean_natures(existing: dict[str, str] | None = None) -> dict[str, str]:
    """Stat Alignment names, from the pkmnchamps dex.

    PokeAPI has these too, but under the old Nature wording; Champions renamed
    the mechanic and pkmnchamps is the surface that shows the game's own. Only
    the 21 alignments the ruleset allows are kept -- the four neutral extras
    exist in the data and cannot be chosen.

    ``existing`` is what is already in the committed file, returned unchanged
    when the raw scrape is not on this machine. Regenerating names.json without
    ``data/raw/pkmnchamps`` must not silently drop a section.
    """
    from pkcm.engine.stats import NATURES

    source = PKMNCHAMPS_DIR / "names_ko.json"
    if not source.exists():
        print("  (no data/raw/pkmnchamps -- run scripts/fetch_pkmnchamps.py; "
              "keeping the committed Stat Alignment names)")
        return dict(existing or {})
    raw = json.loads(source.read_text(encoding="utf-8"))["natures"]
    return {key: entry["ko"] for key, entry in raw.items()
            if key in NATURES and entry.get("ko")}


def overlay_pokechams(dex, species: dict, moves: dict, abilities: dict,
                      items: dict) -> dict[str, list[str]]:
    """Fill the names PokeAPI has none for, from the 포케챔스 dex.

    **Gaps only -- it never overrules a name that is already there.** The
    temptation is to let it win outright, since it is a Champions dex and
    PokeAPI is not, but the two disagree about wording as often as about facts:
    포케챔스 calls Aegislash 킬가르도(실드폼) where every other surface says
    킬가르도, and pkmnchamps (which ``fetch_pkmnchamps.py`` scrapes for the
    party sheets) disagrees with both. Picking a winner among third-party
    spellings is not this file's job; having *a* name for everything is.

    What it does fix is real: ``eelevate`` and ``firemane`` are Champions
    originals that PokeAPI has never heard of, and docs/RESUME.md has carried
    them as an open question -- "no published Korean name, shown in English".
    There is one, and this is where it was.

    Returns what changed, so the diff is visible rather than silent.
    """
    from build_champions_learnsets import our_species_id

    if not POKECHAMS_DIR.exists():
        print("  (no data/raw/pokechams -- run scripts/fetch_pokechams.py; "
              "keeping the composed names)")
        return {}

    def read(name):
        return json.loads((POKECHAMS_DIR / name).read_text(encoding="utf-8"))

    def ident(value: str) -> str:
        return "".join(c for c in (value or "").lower() if c.isalnum())

    changed: dict[str, list[str]] = {}
    known = set(dex.species)
    for entry in read("champions_pokemon.json"):
        species_id = our_species_id(entry["slug"], known)
        korean = entry.get("nameKo")
        if not species_id or not korean or species.get(species_id):
            continue
        changed.setdefault("species", []).append(
            f"{species_id} had no Korean name -> {korean!r}")
        species[species_id] = korean

    for file_name, table, lookup, label in (
        ("moves.json", moves, dex.moves, "moves"),
        ("items.json", items, dex.items, "items"),
        ("abilities.json", abilities, dex.abilities, "abilities"),
    ):
        for entry in read(file_name):
            key = ident(entry.get("nameEn"))
            korean = entry.get("nameKo")
            if key not in lookup or not korean or table.get(key):
                continue
            changed.setdefault(label, []).append(
                f"{key} had no Korean name -> {korean!r}")
            table[key] = korean
    return changed


def main() -> int:
    from pkcm.data.dex import load_dex

    dex = load_dex()

    species_base = names_by_identifier("pokemon_species.csv", "pokemon_species_names.csv",
                                       "pokemon_species_id")
    species, missing_species = korean_species(species_base, dex)
    moves = names_by_identifier("moves.csv", "move_names.csv", "move_id")
    abilities = names_by_identifier("abilities.csv", "ability_names.csv", "ability_id")
    items = names_by_identifier("items.csv", "item_names.csv", "item_id")
    # PokeAPI has never heard of Champions' new Mega Stones, but hk's op.gg
    # scrape carries their Korean names verbatim. That file wins where it has
    # an opinion -- it is the live game's own wording.
    scraped = json.loads(
        (ROOT / "data" / "champions" / "items_m_b.json").read_text(encoding="utf-8")
    )
    for entry in scraped["items"]:
        items[entry["id"]] = entry["korean"]
    types = names_by_identifier("types.csv", "type_names.csv", "type_id")
    committed = (json.loads(OUT_PATH.read_text(encoding="utf-8"))
                 if OUT_PATH.exists() else {})
    natures = korean_natures(committed.get("natures"))

    corrected = overlay_pokechams(dex, species, moves, abilities, items)

    payload = {
        "language": "ko",
        "source": ("PokeAPI multilingual CSVs; formes composed from the base "
                   "name; gaps filled from the 포케챔스 dex"),
        "note": (
            "Ids are never localized -- they key into Showdown's data and the "
            "Champions overrides. Only display names live here."
        ),
        "species": species,
        "moves": moves,
        "abilities": abilities,
        "items": items,
        "types": types,
        "natures": natures,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=0, ensure_ascii=False, sort_keys=True),
                        encoding="utf-8")

    for label, rows in corrected.items():
        print(f"  포케챔스 filled {len(rows)} missing {label}; e.g. {rows[0]}")

    regulation = dex.regulation("m_b")
    roster = regulation.legal_species | regulation.legal_megas
    champions_moves = {m.id for m in dex.moves.values() if dex.exists_in_champions(m)}

    def coverage(label, wanted, table):
        have = len(wanted & set(table))
        gap = sorted(wanted - set(table))
        print(f"  {label:12} {have}/{len(wanted)}"
              + (f"   missing: {gap[:6]}{' ...' if len(gap) > 6 else ''}" if gap else ""))
        return gap

    print("Korean coverage for what Champions actually uses:")
    coverage("species", roster, species)
    coverage("moves", champions_moves, moves)
    coverage("abilities", {a for s in roster for a in dex.species[s].abilities}, abilities)
    from pkcm.engine.items import champions_items

    coverage("items", champions_items(), items)
    print(f"\nwritten -> {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
