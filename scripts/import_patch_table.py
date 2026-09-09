"""Read the M-C changes table into JSON, and say where it disagrees with us.

hk was handed a table someone else compiled from the game's v1.2.0 data and
checked it themselves. It is the first Champions-native source this project
has had for base stats, abilities and learnsets all at once -- everything the
dex holds today came from play.pokemonshowdown.com, which describes a
different game.

The comparison is the point. Showdown had Samurott as a Water type with
95/100/85/108/70/70 and Torrent; Champions has it Bug/Water with
75/125/140/60/90/40 and Emergency Exit. Nothing about the dex's numbers can be
assumed until this has been run over all of them.

Usage:
    python scripts/import_patch_table.py <saved.html> [--out data/champions/mc_table.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

NAMES = json.loads((ROOT / "data" / "champions" / "names.json")
                   .read_text(encoding="utf-8"))

#: Korean type names to the ids the engine uses.
TYPES = {"노말": "normal", "불꽃": "fire", "물": "water", "전기": "electric",
         "풀": "grass", "얼음": "ice", "격투": "fighting", "독": "poison",
         "땅": "ground", "비행": "flying", "에스퍼": "psychic", "벌레": "bug",
         "바위": "rock", "고스트": "ghost", "드래곤": "dragon", "악": "dark",
         "강철": "steel", "페어리": "fairy"}


#: The table writes some names differently from ``names.json``, which was
#: built from PokeAPI's CSVs. Formes are spelt out where the name table
#: abbreviates, the Z Megas need the suffix to be told from the ordinary ones
#: -- both are just "메가앱솔" in the name table -- and two moves are in the
#: game's noun form against the CSV's verb form.
ALIASES = {
    "species": {
        "페르시온 (알로라의 모습)": "persianalola",
        "메가앱솔Z": "absolmegaz",
        "메가한카리아스Z": "garchompmegaz",
        "메가루카리오Z": "lucariomegaz",
        "스트린더 (하이한 모습)": "toxtricity",
        "스트린더 (로우한 모습)": "toxtricitylowkey",
        "에써르 (수컷)": "indeedee",
        "에써르 (암컷)": "indeedeef",
    },
    "moves": {"깨트리기": "brickbreak", "탐내기": "covet"},
    "abilities": {"파동의방호": "auraguard"},
}


def plain_text(html: str) -> str:
    """The page as lines. Its own markup is the only structure it has."""
    body = re.sub(r"<script.*?</script>|<style.*?</style>", "", html, flags=re.S)
    body = re.sub(r"</(td|th|tr|div|p|li|h[1-6]|section)>", "\n", body)
    body = re.sub(r"<[^>]+>", " ", body)
    body = re.sub(r"[ \t]+", " ", body)
    return "\n".join(line.strip() for line in body.split("\n") if line.strip())


STAT_LINE = re.compile(r"^(\d+) / (\d+) / (\d+) / (\d+) / (\d+) / (\d+) = (\d+)$")


def parse(text: str) -> dict:
    """Walk the lines, because the page is a sequence rather than a table.

    A species is a name, a type line, a stat line and then either abilities
    with a Mega Stone -- which is how a Mega is written -- or abilities and
    two move lists.
    """
    lines = text.split("\n")
    species: list[dict] = []
    for index, line in enumerate(lines):
        stats = STAT_LINE.match(line)
        if not stats or index < 2:
            continue
        types = [TYPES[word] for word in lines[index - 1].split()
                 if word in TYPES]
        if not types:
            continue
        entry = {
            "ko": lines[index - 2],
            "types": types,
            "base": dict(zip(("hp", "atk", "def", "spa", "spd", "spe"),
                             (int(stats.group(i)) for i in range(1, 7)))),
            "total": int(stats.group(7)),
        }
        for ahead in lines[index + 1:index + 6]:
            if ahead.startswith("특성 "):
                # A Mega writes "특성 <one> · <its stone>"; a species writes
                # "특성 <a> / <b> / <c>".
                rest = ahead[3:]
                if " · " in rest and " / " not in rest:
                    ability, _, stone = rest.partition(" · ")
                    entry["abilities_ko"] = [ability.strip()]
                    entry["stone_ko"] = stone.strip()
                else:
                    entry["abilities_ko"] = [one.strip()
                                             for one in rest.split(" / ")]
                break
        moves: list[str] = []
        for ahead in lines[index + 1:index + 8]:
            if ahead.startswith("공격 ") or ahead.startswith("변화 "):
                moves += [one.strip() for one in ahead[3:].split(" · ")]
        if moves:
            entry["moves_ko"] = moves
        species.append(entry)
    return {"species": species}


def resolve(entry: dict) -> dict:
    """Korean display names to the ids everything else keys on."""
    back = {kind: {name: key for key, name in table.items()}
            for kind, table in NAMES.items() if isinstance(table, dict)}
    for kind, extra in ALIASES.items():
        back.setdefault(kind, {}).update(extra)
    entry["id"] = back["species"].get(entry["ko"])
    entry["abilities"] = [back["abilities"].get(one)
                          for one in entry.get("abilities_ko", ())]
    if "moves_ko" in entry:
        entry["moves"] = [back["moves"].get(one) for one in entry["moves_ko"]]
    if "stone_ko" in entry:
        entry["stone"] = back["items"].get(entry["stone_ko"])
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    text = plain_text(Path(args.html).read_text(encoding="utf-8",
                                                errors="replace"))
    table = parse(text)
    for entry in table["species"]:
        resolve(entry)

    print(f"{len(table['species'])} species read")
    unknown = [e["ko"] for e in table["species"] if e["id"] is None]
    if unknown:
        print(f"  no id for: {', '.join(unknown)}")
    missing_moves = sorted({ko for e in table["species"]
                            for ko, mid in zip(e.get("moves_ko", ()),
                                               e.get("moves", ()))
                            if mid is None})
    if missing_moves:
        print(f"  no id for {len(missing_moves)} moves: "
              f"{', '.join(missing_moves[:12])}")
    missing_ab = sorted({ko for e in table["species"]
                         for ko, aid in zip(e.get("abilities_ko", ()),
                                            e.get("abilities", ()))
                         if aid is None})
    if missing_ab:
        print(f"  no id for abilities: {', '.join(missing_ab)}")

    if args.out:
        Path(args.out).write_text(
            json.dumps(table, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        print(f"  written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
