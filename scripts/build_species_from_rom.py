"""Build the species table out of the game's own personal table.

**Why this exists.** Showdown was never agreed to be a source of facts about
Champions -- it is a reference for how mechanics are structured -- and every
time species data was taken from it anyway, it eventually said something the
game does not. The last time cost a day: Mega Golisopod loaded with Emergency
Exit instead of Tough Claws, because Showdown cannot know the abilities of a
forme that exists only in Champions and falls back to the base species'.

So the facts come from ``champout/personal.json``, which is the ROM's own
table, and Showdown is used for three dictionaries and nothing else:

* type index to type name. Eighteen entries, and every one of them resolves to
  exactly one Showdown type across every base forme, so the ordering is read
  off the data rather than assumed.
* ability number to ability id, through the base data's ``num`` field. The
  mainline numbering runs to 318; Champions-original abilities above that are
  named in ``rom_abilities.json`` instead.
* dex number and forme index to the key the rest of this codebase uses,
  through ``formeOrder``.

None of the three is a statement about what Champions contains. They are how a
number is spelled.

**What it checks.** Mapped by dex number and forme index alone -- never by
stats, which would make the comparison circular -- the ROM and Showdown agree
on the types and base stats of every mapped forme, and disagree on the
abilities of exactly five: the Champions-original Megas. That is the shape of
the problem, measured rather than assumed, and it is printed on every run so
the day it changes is a day somebody notices.

Usage:
    python scripts/build_species_from_rom.py [--check]
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
OUT = CHAMPIONS / "species.json"

#: The personal table's type column. Verified rather than assumed: across
#: every base forme, each index resolves to exactly one Showdown type.
TYPES = ("Normal", "Fighting", "Flying", "Poison", "Ground", "Rock", "Bug",
         "Ghost", "Steel", "Fire", "Water", "Grass", "Electric", "Psychic",
         "Ice", "Dragon", "Dark", "Fairy")


def to_id(text: str) -> str:
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build() -> tuple[dict, dict]:
    """The species table, and a report of where the ROM and Showdown differ."""
    rom = load(DUMP / "personal.json")
    pokedex = load(RAW / "pokedex.json")
    abilities = load(RAW / "abilities.json")
    added = load(CHAMPIONS / "overrides.json")["abilities"].get("added", {})
    extra = load(CHAMPIONS / "rom_abilities.json")["names"]

    by_num = {value["num"]: key for key, value in abilities.items()
              if isinstance(value.get("num"), int)}
    by_num.update({int(num): name for num, name in extra.items()})
    named = {key: value.get("name", key) for key, value in abilities.items()}
    named.update({key: entry.get("name", key) for key, entry in added.items()})

    base_of_num = {value["num"]: key for key, value in pokedex.items()
                   if isinstance(value.get("num"), int)
                   and not value.get("forme") and value.get("baseStats")}

    table: dict[str, dict] = {}
    report: dict[str, list] = {"unmapped": [], "type_differs": [],
                               "stats_differ": [], "abilities_differ": [],
                               "weight_differs": [], "unnamed_ability": []}

    for entry in rom:
        if entry.get("is_valid") != "1":
            continue
        base = base_of_num.get(int(entry["no"]))
        if base is None:
            report["unmapped"].append((entry["id"], "no such dex number"))
            continue
        order = pokedex[base].get("formeOrder") or [pokedex[base]["name"]]
        index = int(entry["fo"])
        if index >= len(order):
            report["unmapped"].append((entry["id"], f"forme {index} of {order}"))
            continue
        key = to_id(order[index])
        if key not in pokedex or not pokedex[key].get("baseStats"):
            report["unmapped"].append((entry["id"], f"{key} not in the base data"))
            continue

        first, second = TYPES[int(entry["type1"])], TYPES[int(entry["type2"])]
        types = [first] if first == second else [first, second]
        stats = {"hp": int(entry["hp"]), "atk": int(entry["atk"]),
                 "def": int(entry["def"]), "spa": int(entry["spatk"]),
                 "spd": int(entry["spdef"]), "spe": int(entry["agi"])}

        # The personal table's three ability columns are the two ordinary
        # slots and the hidden one, in that order, and a species with fewer
        # than three distinct abilities repeats an earlier column rather than
        # leaving a hole. The base data names those slots "0", "1" and "H" --
        # not "0", "1", "2" -- and ``_build_species`` reads exactly those
        # keys, so writing the wrong ones silently drops every hidden ability
        # in the game. It did: 86 of 243 field parties turned illegal.
        slots: dict[str, str] = {}
        ordered: list[str] = []
        for column, slot in (("toku0", "0"), ("toku1", "1"), ("toku2", "H")):
            number = int(entry[column])
            if not number:
                continue
            ability = by_num.get(number)
            if ability is None:
                report["unnamed_ability"].append((key, number))
                continue
            if ability in ordered:
                continue
            ordered.append(ability)
            slots[slot] = ability

        # Hectograms in the table, kilograms in the engine. Weight is a
        # battle-relevant fact -- Low Kick, Grass Knot, Heavy Slam and Heat
        # Crash all read it -- so it comes from here too rather than being
        # left behind as the one species number Showdown still owned.
        weight = int(entry["weight"]) / 10.0

        held = pokedex[key]
        if abs(float(held.get("weightkg", 0.0)) - weight) > 1e-9:
            report["weight_differs"].append(
                (key, held.get("weightkg"), weight))
        if list(held["types"]) != types:
            report["type_differs"].append((key, list(held["types"]), types))
        if {k: held["baseStats"][k] for k in stats} != stats:
            report["stats_differ"].append((key, held["baseStats"], stats))
        theirs = [to_id(name) for _, name in
                  sorted((held.get("abilities") or {}).items())]
        if theirs != ordered:
            report["abilities_differ"].append((key, theirs, ordered))

        table[key] = {
            "rom_id": entry["id"],
            "num": int(entry["no"]),
            "types": types,
            "baseStats": stats,
            "weightkg": weight,
            "abilities": {slot: named.get(a, a) for slot, a in slots.items()},
        }
    return table, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="write nothing; exit 1 if species.json is stale")
    args = parser.parse_args()

    table, report = build()
    if report["unnamed_ability"]:
        for key, number in sorted(set(report["unnamed_ability"])):
            print(f"  {key}: ability {number} has no name anywhere. Add it to "
                  f"data/champions/rom_abilities.json")
        return 2

    if args.check:
        if not OUT.exists():
            print(f"{OUT} is missing. Run scripts/build_species_from_rom.py")
            return 1
        if load(OUT).get("species") != table:
            print("species.json is stale. Run scripts/build_species_from_rom.py")
            return 1
        print(f"species.json up to date ({len(table)} formes)")
        return 0

    payload = {
        "source": ("data/raw/champout/personal.json -- the game's own table. "
                   "Showdown supplies no fact here, only the spelling of a "
                   "type index, an ability number and a forme index. Rebuild "
                   "with scripts/build_species_from_rom.py."),
        "species": table,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")

    print(f"wrote {len(table)} formes to {OUT}")
    print(f"  unmapped ROM entries  : {len(report['unmapped'])}"
          f"   (cosmetic formes the base data does not split)")
    print(f"  types the ROM corrects: {len(report['type_differs'])}")
    print(f"  stats the ROM corrects: {len(report['stats_differ'])}")
    print(f"  weights it corrects   : {len(report['weight_differs'])}")
    for key, theirs, ours in sorted(report["weight_differs"])[:20]:
        print(f"      {key:22} {theirs} -> {ours}")
    print(f"  abilities it corrects : {len(report['abilities_differ'])}")
    for key, theirs, ours in sorted(report["abilities_differ"]):
        print(f"      {key:22} {','.join(theirs) or '-'} -> {','.join(ours) or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
