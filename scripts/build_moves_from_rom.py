"""Build the move table's numbers out of the game's own waza table.

The companion to ``build_species_from_rom.py`` and the same argument: what a
move's type, category, power and accuracy *are* is a fact about Champions,
and facts come from the ROM.

**What it changes and what it does not.** Only the four numbers above. A move's
*behaviour* -- what Encore does, which flags it carries, what its secondary
effect is -- stays with the base data and the engine's own handlers, because
that is the thing Showdown is genuinely a reference for. The ROM's own effect
columns (``classification_a``, ``con_ref``, ``buf_ref``) are opaque indices
into tables this dump does not include.

**PP is deliberately not in here.** The waza table's ``pp`` column is *fully
boosted* PP -- what the move has in a battle after Champions' own formula,
``(base // 5 + 1) * 4`` on a base capped at 20 -- and not base PP. Read as base
PP it looks like four fifths of the move list is wrong, which is what it looked
like to me for about ten minutes. Read correctly it agrees with what this
engine already computes on 502 of 512 moves, and the ten that differ are
already corrected in the move overrides. So PP is checked in
``tests/test_champions_rules.py`` against ``max_pp`` and overridden nowhere.

Usage:
    python scripts/build_moves_from_rom.py [--check]
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

CHAMPIONS = ROOT / "data" / "champions"
RAW = ROOT / "data" / "raw"
DUMP = RAW / "champout"
OUT = CHAMPIONS / "moves.json"

TYPES = ("Normal", "Fighting", "Flying", "Poison", "Ground", "Rock", "Bug",
         "Ghost", "Steel", "Fire", "Water", "Grass", "Electric", "Psychic",
         "Ice", "Dragon", "Dark", "Fairy")
CATEGORIES = {"0": "Physical", "1": "Special", "2": "Status"}

#: The table writes 1 for a move whose damage is not its base power at all --
#: Seismic Toss, Counter, Super Fang, the OHKO moves, Flail. The base data
#: writes 0 for the same thing and the engine reads 0, so the marker is left
#: alone rather than translated into a one-power attack.
FIXED_DAMAGE_MARKER = 1

#: The table writes 101 for a move that cannot miss -- one past the top of the
#: percentage scale, which is how it says "no roll". The base data writes true.
ALWAYS_HITS = 101


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build() -> tuple[dict, dict]:
    waza = load(DUMP / "waza.json")
    moves = load(RAW / "moves.json")
    by_num = {value["num"]: key for key, value in moves.items()
              if isinstance(value.get("num"), int)}

    table: dict[str, dict] = {}
    report: dict[str, list] = {"unmapped": [], "type": [], "category": [],
                               "power": [], "accuracy": []}

    for entry in waza:
        if entry.get("available") != "1":
            continue
        key = by_num.get(int(entry["id"]))
        if key is None:
            report["unmapped"].append(entry["id"])
            continue
        held = moves[key]

        fields: dict[str, object] = {
            "type": TYPES[int(entry["type"])],
            "category": CATEGORIES[entry["category"]],
        }
        power = int(entry["power"])
        if power != FIXED_DAMAGE_MARKER:
            fields["basePower"] = power
        accuracy = int(entry["accuracy"])
        fields["accuracy"] = True if accuracy == ALWAYS_HITS else accuracy

        if held.get("type") != fields["type"]:
            report["type"].append((key, held.get("type"), fields["type"]))
        if held.get("category") != fields["category"]:
            report["category"].append((key, held.get("category"), fields["category"]))
        if "basePower" in fields and held.get("basePower") != fields["basePower"]:
            report["power"].append((key, held.get("basePower"), fields["basePower"]))
        if held.get("accuracy") != fields["accuracy"]:
            report["accuracy"].append((key, held.get("accuracy"), fields["accuracy"]))
        table[key] = fields
    return table, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="write nothing; exit 1 if moves.json is stale")
    args = parser.parse_args()

    table, report = build()

    if args.check:
        if not OUT.exists():
            print(f"{OUT} is missing. Run scripts/build_moves_from_rom.py")
            return 1
        if load(OUT).get("moves") != table:
            print("moves.json is stale. Run scripts/build_moves_from_rom.py")
            return 1
        print(f"moves.json up to date ({len(table)} moves)")
        return 0

    OUT.write_text(json.dumps(
        {"source": ("data/raw/champout/waza.json -- the game's own table. "
                    "Type, category, power and accuracy only -- not PP, "
                    "whose column is boosted rather than base. Behaviour "
                    "stays with the engine's handlers. Rebuild with "
                    "scripts/build_moves_from_rom.py."),
         "moves": table}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")

    print(f"wrote {len(table)} moves to {OUT}")
    print(f"  unmapped     : {len(report['unmapped'])}")
    for field in ("type", "category", "power", "accuracy"):
        rows = report[field]
        print(f"  {field:13}: {len(rows)} differ from the base data")
        for row in rows[:8]:
            print(f"      {row[0]:22} {row[1]} -> {row[2]}")
        if len(rows) > 8:
            print(f"      ... and {len(rows) - 8} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
