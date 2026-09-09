"""Read the M-C changes table into JSON, and say where it disagrees with us.

hk was handed a table someone else compiled from the game's v1.2.0 data and
checked it themselves. It is the first Champions-native source this project
has had for base stats, abilities and learnsets all at once -- everything the
dex holds today came from play.pokemonshowdown.com, which describes a
different game.

The comparison is the point, and it came out narrower than expected: every
type and every base stat matched, and the only abilities that did not were the
five Megas Champions invented, which Showdown cannot know about. (The example
that used to sit here said Champions had rewritten Samurott. It had not --
갑주무사 is Golisopod, and I had the name wrong.)

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
        # The name table calls the Shield forme just 킬가르도 and names only
        # the Blade one, which is the opposite of how the table writes it.
        "킬가르도 (실드폼)": "aegislash",
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
    return {"species": species, **parse_changes(lines)}


POWER = re.compile(r"^(\d+) → (\d+)$")
#: The table writes removals with a minus sign, not a hyphen.
DROPPED = re.compile(r"^(.+?) − (.+)$")
SPREAD = re.compile(r"^(.+?) 추가 \((\d+)종\) (.+)$")


def parse_changes(lines: list[str]) -> dict:
    """The three sections above the species: powers, the format, learnsets.

    Separate from ``parse`` because they are lists rather than blocks, and
    because the learnset section is the one thing here that no other source
    has: which *existing* species gained or lost a move.
    """
    def between(start: str, *stops: str) -> list[str]:
        if start not in lines:
            return []
        head = lines.index(start) + 1
        tail = min((lines.index(one) for one in stops if one in lines),
                   default=len(lines))
        return lines[head:tail]

    powers = {}
    section = between("위력 변경", "사용 가능 기술")
    for name, value in zip(section, section[1:]):
        shift = POWER.match(value)
        if shift:
            powers[name] = [int(shift.group(1)), int(shift.group(2))]

    added, removed = [], []
    for line in between("사용 가능 기술", "기존 포켓몬 기술 변경"):
        if line.startswith("추가 "):
            added = line.split(" ", 2)[2].split(" · ")
        elif line.startswith("삭제 "):
            removed = line[3:].split(" · ")

    gained: dict[str, list[str]] = {}
    lost: dict[str, list[str]] = {}
    for line in between("기존 포켓몬 기술 변경", "신규 포켓몬 32"):
        spread = SPREAD.match(line)
        if spread:
            move = spread.group(1)
            for who in spread.group(3).split(" · "):
                gained.setdefault(who.strip(), []).append(move)
            continue
        drop = DROPPED.match(line)
        if drop:
            lost[drop.group(1).strip()] = [one.strip()
                                           for one in drop.group(2).split(" · ")]

    return {"power_changes_ko": powers,
            "format_moves_ko": {"added": added, "removed": removed},
            "existing_species_ko": {"gained": gained, "lost": lost}}


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


def resolve_changes(table: dict) -> None:
    """Ids for the change lists, keeping the Korean beside them."""
    back = {kind: {name: key for key, name in one.items()}
            for kind, one in NAMES.items() if isinstance(one, dict)}
    for kind, extra in ALIASES.items():
        back.setdefault(kind, {}).update(extra)

    table["power_changes"] = {back["moves"].get(ko, ko): value
                              for ko, value in table["power_changes_ko"].items()}
    table["format_moves"] = {
        which: [back["moves"].get(ko) for ko in names]
        for which, names in table["format_moves_ko"].items()}
    table["existing_species"] = {
        which: {back["species"].get(who): [back["moves"].get(mv) for mv in moves]
                for who, moves in group.items()}
        for which, group in table["existing_species_ko"].items()}


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
    resolve_changes(table)

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

    print(f"{len(table['power_changes'])} power changes, "
          f"{len(table['format_moves']['added'])} moves added to the format, "
          f"{len(table['format_moves']['removed'])} removed")
    gained, lost = table["existing_species"]["gained"], table["existing_species"]["lost"]
    print(f"{len(gained)} existing species gain a move, {len(lost)} lose one")
    for which, group, korean in (("gained", gained, table["existing_species_ko"]["gained"]),
                                 ("lost", lost, table["existing_species_ko"]["lost"])):
        nameless = [ko for ko, who in zip(korean, group) if who is None]
        if nameless:
            print(f"  no id for {which}: {', '.join(nameless)}")
    unknown_moves = sorted({mv for group in (gained, lost)
                            for moves in group.values() for mv in moves
                            if mv is None})
    if unknown_moves or None in table["format_moves"]["added"]             or None in table["format_moves"]["removed"]:
        print("  some move names did not resolve")

    if args.out:
        Path(args.out).write_text(
            json.dumps(table, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        print(f"  written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
