"""Correct the pokedex where Showdown cannot know what Champions did.

Showdown's Champions mod is a good source for a species' types and base stats
-- on the 2026-09-09 update it was exact on 32 of 32 for both, checked against
the game's own table. It is not a source for the *abilities* of a forme
Champions invented. There is no mainline Mega Golisopod, so there is no
ability for Showdown to carry, and it falls back to the base species': Mega
Golisopod arrived with Emergency Exit instead of Tough Claws, Mega Lucario Z
with Adaptability instead of the Aura Guard this engine has an implementation
for, Mega Absol Z with Magic Bounce instead of Sharpness, Mega Garchomp Z with
Sand Force instead of Levitate, and Mega Baxcalibur with an Ice Body it does
not have alongside its Thermal Exchange.

All five were written down in patch_2026_09_09.json the day they were learned
and none of them reached the dex, because there was no species override layer
to put them in. This builds one.

The corrections are read out of mc_table.json rather than typed here, so the
file the game's data lives in stays the only place they are written. Run this
after any change to that table.

Usage:
    python scripts/build_species_overrides.py [--check]

``--check`` writes nothing and exits non-zero if the overrides are stale,
which is what the test uses.
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


def to_id(text: str) -> str:
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def wanted() -> dict[str, dict]:
    """Every species change the game's table implies, as override entries.

    Only abilities. Types and base stats are not touched: they were checked
    against the same table and matched everywhere, so writing them would be
    restating Showdown's own numbers back at it and would hide the day they
    stop agreeing.
    """
    table = json.loads((CHAMPIONS / "mc_table.json").read_text(encoding="utf-8"))
    pokedex = json.loads((RAW / "pokedex.json").read_text(encoding="utf-8"))
    changes: dict[str, dict] = {}
    for entry in table["species"]:
        base = pokedex.get(entry["id"])
        if base is None:
            continue
        # The table lists an ability twice for a couple of species; the dex
        # slots are positional, so duplicates are dropped in order rather than
        # sorted away.
        seen, ordered = set(), []
        for ability in entry.get("abilities") or ():
            if ability not in seen:
                seen.add(ability)
                ordered.append(ability)
        if not ordered:
            continue
        have = [to_id(name) for _, name in
                sorted((base.get("abilities") or {}).items())]
        if have == ordered:
            continue
        names = {}
        for index, ability in enumerate(ordered):
            names[str(index)] = _display_name(ability)
        changes[entry["id"]] = {"abilities": names}
    return changes


def _display_name(ability_id: str) -> str:
    """The name Showdown's own ability table uses, so the dex resolves it.

    ``_build_species`` runs ``to_id`` over whatever is here, so the exact
    spelling does not matter to the engine -- but an ability that is not in
    the table at all is a typo we want to hear about now rather than in a
    battle.
    """
    abilities = json.loads((RAW / "abilities.json").read_text(encoding="utf-8"))
    # Including the ones Champions added after the mod data was captured,
    # which live only in the override file -- Aura Guard is the first.
    overrides = json.loads((CHAMPIONS / "overrides.json").read_text(encoding="utf-8"))
    abilities.update(overrides["abilities"].get("added", {}))
    entry = abilities.get(ability_id)
    if entry is None:
        raise SystemExit(f"no ability called {ability_id!r} in the base data; "
                         "either it is misspelled in mc_table.json or it needs "
                         "adding to the ability overrides first")
    return entry.get("name", ability_id)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="write nothing; exit 1 if the overrides are stale")
    args = parser.parse_args()

    path = CHAMPIONS / "overrides.json"
    overrides = json.loads(path.read_text(encoding="utf-8"))
    changes = wanted()
    held = overrides.get("species", {}).get("changes", {})

    if args.check:
        if held != changes:
            missing = sorted(set(changes) - set(held))
            stale = sorted(key for key in changes
                           if key in held and held[key] != changes[key])
            extra = sorted(set(held) - set(changes))
            print("species overrides are stale. Run "
                  "scripts/build_species_overrides.py")
            for label, keys in (("missing", missing), ("changed", stale),
                                ("no longer needed", extra)):
                if keys:
                    print(f"  {label}: {', '.join(keys)}")
            return 1
        print(f"species overrides up to date ({len(changes)} entries)")
        return 0

    overrides["species"] = {
        "source": ("data/champions/mc_table.json -- the game's own v1.2.0 "
                   "table as hk checked it. Showdown carries the base "
                   "species' abilities for formes Champions invented, which "
                   "is the only thing corrected here."),
        "changes": changes,
    }
    path.write_text(json.dumps(overrides, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8")
    print(f"wrote {len(changes)} species overrides to {path}")
    for key, fields in sorted(changes.items()):
        print(f"  {key}: {', '.join(fields['abilities'].values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
