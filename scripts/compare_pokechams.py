"""Diff our dex against the 포케챔스 snapshot, and say where they disagree.

Two independent readings of the same game: ours is built from Showdown's
champions mod, theirs from the game's own dex. Neither is the answer key --
the game is -- so this prints the disagreements rather than resolving them.

Species match on their ``slug``, which is our id with hyphens. Moves, abilities
and items match on **English name**: the two sources number things differently
and Champions renumbered about thirty moves on top of that, so an id match
would be a coincidence while a name match is a fact.

Usage:
    python scripts/compare_pokechams.py              # every section, summarised
    python scripts/compare_pokechams.py --full       # every row, not just a sample
    python scripts/compare_pokechams.py --only stats # one section
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pkcm.data.dex import Stat, load_dex  # noqa: E402

RAW_DIR = ROOT / "data" / "raw" / "pokechams"

#: How many rows a section prints before it starts counting instead.
SAMPLE = 15


def normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def load(filename: str) -> list[dict]:
    path = RAW_DIR / filename
    if not path.exists():
        raise SystemExit(f"missing {path}\nrun: python scripts/fetch_pokechams.py")
    return json.loads(path.read_text(encoding="utf-8"))


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def rows(label: str, items: list[str], full: bool) -> None:
    print(f"\n  {label} ({len(items)})")
    if not items:
        print("    none")
        return
    shown = items if full else items[:SAMPLE]
    for item in shown:
        print(f"    {item}")
    if len(items) > len(shown):
        print(f"    ... and {len(items) - len(shown)} more  (--full to see them)")


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #

#: Kept in step with ``scripts/build_champions_learnsets.py``. Duplicated rather
#: than imported because that script is a build step and this one is a report;
#: a test asserts the two agree.
SLUG_ALIASES = {
    "meowstic-female": "meowsticf",
    "meowstic-mega": "meowsticmmega",
    "lycanroc-midday": "lycanroc",
    "basculegion-female": "basculegionf",
}


def our_species_id(slug: str, known: set[str]) -> str | None:
    if slug in SLUG_ALIASES:
        return SLUG_ALIASES[slug]
    candidates = [slug.replace("-", "")]
    if "-mega-" in slug:
        candidates.append(slug.split("-mega-")[0].replace("-", "") + "mega")
    if slug.endswith("-breed"):
        candidates.append(slug[: -len("-breed")].replace("-", ""))
    return next((c for c in candidates if c in known), None)


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #

def compare_roster(dex, theirs, regulation, full: bool) -> dict[str, str]:
    """Returns their-slug -> our-id for everything that matched."""
    section("ROSTER  -- which Pokemon are in Champions at all")

    known = set(dex.species)
    matched: dict[str, str] = {}
    unmatched_slugs: list[str] = []

    for entry in theirs:
        if not entry.get("allowedInChampions"):
            continue
        ours = our_species_id(entry["slug"], known)
        if ours is None:
            unmatched_slugs.append(f"{entry['slug']:34} {entry['nameEn']}")
        else:
            matched[entry["slug"]] = ours

    ours_legal = regulation.legal_species | regulation.legal_megas
    print(f"\n  theirs: {len(matched) + len(unmatched_slugs):>4} allowed"
          f"   ours: {len(ours_legal):>4} legal")

    rows("their slugs we cannot resolve to a species we know", unmatched_slugs, full)

    theirs_as_ours = set(matched.values())
    rows("they allow, we do not list as legal",
         [f"{s:24} {dex.species[s].name}" for s in sorted(theirs_as_ours - ours_legal)], full)
    rows("we list as legal, they do not allow",
         [f"{s:24} {dex.species[s].name}" for s in sorted(ours_legal - theirs_as_ours)], full)
    return matched


def compare_species_fields(dex, theirs, matched, full: bool) -> None:
    section("SPECIES DATA  -- base stats, types, abilities, weight")

    by_slug = {entry["slug"]: entry for entry in theirs}
    stat_rows: list[str] = []
    type_rows: list[str] = []
    ability_rows: list[str] = []
    weight_rows: list[str] = []

    order = ("hp", "atk", "def", "spa", "spd", "spe")
    for slug, species_id in sorted(matched.items(), key=lambda kv: kv[1]):
        theirs_entry = by_slug[slug]
        ours = dex.species[species_id]

        mine = tuple(ours.base_stats[getattr(Stat, name.upper())] for name in order)
        yours = tuple(theirs_entry["baseStats"][name] for name in order)
        if mine != yours:
            stat_rows.append(f"{ours.name:24} ours={mine}  theirs={yours}")

        mine_types = tuple(sorted(t.lower() for t in ours.types))
        their_types = tuple(sorted(t.lower() for t in theirs_entry["types"]))
        if mine_types != their_types:
            type_rows.append(f"{ours.name:24} ours={mine_types}  theirs={their_types}")

        if abs(float(ours.weight_kg) - float(theirs_entry["weightKg"])) > 0.05:
            weight_rows.append(
                f"{ours.name:24} ours={ours.weight_kg}kg  theirs={theirs_entry['weightKg']}kg")

    rows("base stat disagreements", stat_rows, full)
    rows("type disagreements", type_rows, full)
    rows("weight disagreements", weight_rows, full)
    _compare_abilities_per_species(dex, theirs, matched, full)


def _compare_abilities_per_species(dex, theirs, matched, full: bool) -> None:
    their_abilities = {record["id"]: record["nameEn"] for record in load("abilities.json")}
    ours_by_name = {normalise(a["name"]): key for key, a in _our_ability_names(dex).items()}

    by_slug = {entry["slug"]: entry for entry in theirs}
    diff_rows: list[str] = []
    unmapped: set[str] = set()

    for slug, species_id in sorted(matched.items(), key=lambda kv: kv[1]):
        entry = by_slug[slug]
        their_set = set()
        for ability_id in entry.get("abilities", []):
            name = their_abilities.get(ability_id)
            if name is None:
                unmapped.add(f"id:{ability_id}")
                continue
            ours = ours_by_name.get(normalise(name))
            if ours is None:
                unmapped.add(name)
                continue
            their_set.add(ours)
        our_set = set(dex.species[species_id].abilities)
        if their_set and their_set != our_set:
            diff_rows.append(
                f"{dex.species[species_id].name:24} ours={sorted(our_set)}  "
                f"theirs={sorted(their_set)}")

    rows("ability-slot disagreements", diff_rows, full)
    if unmapped:
        rows("their abilities we could not map", sorted(unmapped), full)


def _our_ability_names(dex) -> dict[str, dict]:
    """ability id -> {'name': display name}, from whatever the dex exposes."""
    table = {}
    for ability_id, entry in dex.abilities.items():
        name = entry.name if hasattr(entry, "name") else entry.get("name", ability_id)
        table[ability_id] = {"name": name}
    return table


def compare_moves(dex, their_moves, full: bool) -> None:
    section("MOVES  -- existence, PP, power, accuracy, priority")

    their_in = {normalise(m["nameEn"]): m for m in their_moves if m.get("allowedInChampions")}
    ours = {normalise(m.name): m for m in dex.moves.values() if dex.exists_in_champions(m)}

    print(f"\n  theirs: {len(their_in):>4} allowed   ours: {len(ours):>4} in Champions")
    rows("only in pokechams", [their_in[k]["nameEn"] for k in sorted(set(their_in) - set(ours))], full)
    rows("only in ours", [ours[k].name for k in sorted(set(ours) - set(their_in))], full)

    from pkcm.engine.pokemon import max_pp

    buckets = {"pp": [], "power": [], "accuracy": [], "priority": []}
    for key in sorted(set(their_in) & set(ours)):
        mine, yours = ours[key], their_in[key]
        if max_pp(mine.pp) != yours["pp"]:
            buckets["pp"].append(f"{mine.name:22} ours={max_pp(mine.pp)}  theirs={yours['pp']}")
        if mine.category != "Status" and mine.base_power and mine.base_power != yours["power"]:
            buckets["power"].append(
                f"{mine.name:22} ours={mine.base_power}  theirs={yours['power']}")
        # Their 101 is our None: "never misses", spelled differently.
        mine_acc = 101 if mine.accuracy is None else mine.accuracy
        if yours["accuracy"] and mine_acc != yours["accuracy"]:
            buckets["accuracy"].append(
                f"{mine.name:22} ours={mine_acc}  theirs={yours['accuracy']}")
        if mine.priority != yours["priority"]:
            buckets["priority"].append(
                f"{mine.name:22} ours={mine.priority}  theirs={yours['priority']}")

    for field, found in buckets.items():
        rows(f"{field} disagreements", found, full)


def compare_catalogue(dex, regulation, full: bool) -> None:
    section("ABILITIES AND ITEMS  -- existence only")

    # Importing the engine is what fills the effect registry. Reading it without
    # that gives an empty answer that reads as "nothing is implemented".
    from pkcm.engine import abilities as _abilities  # noqa: F401
    from pkcm.engine import conditions as _conditions  # noqa: F401
    from pkcm.engine import items as _items  # noqa: F401
    from pkcm.engine import moveeffects as _moveeffects  # noqa: F401
    from pkcm.engine.effects import registered

    names = _our_ability_names(dex)
    their_abilities = {normalise(a["nameEn"]): a["nameEn"] for a in load("abilities.json")}
    # Ours knows every ability Showdown does; only the roster's are comparable.
    roster = {ability
              for species_id in regulation.legal_species | regulation.legal_megas
              for ability in dex.species[species_id].abilities}
    ours_abilities = {normalise(names[a]["name"]): names[a]["name"]
                      for a in roster if a in names}
    print(f"\n  abilities  theirs: {len(their_abilities):>4}"
          f"   ours (roster only): {len(ours_abilities):>4}")
    rows("abilities on our roster that they do not list",
         sorted(ours_abilities[k] for k in set(ours_abilities) - set(their_abilities)), full)
    rows("abilities they list that no roster Pokemon of ours carries",
         sorted(their_abilities[k] for k in set(their_abilities) - set(ours_abilities)), full)

    their_items = {normalise(i["nameEn"]): i["nameEn"] for i in load("items.json")}
    ours_items = {normalise(dex.items[i].name): dex.items[i].name
                  for i in registered("item") if i in dex.items}
    print(f"\n  items      theirs: {len(their_items):>4}"
          f"   ours: {len(ours_items):>4} implemented")
    rows("items they list and we do not implement",
         sorted(their_items[k] for k in set(their_items) - set(ours_items)), full)
    rows("items we implement and they do not list",
         sorted(ours_items[k] for k in set(ours_items) - set(their_items)), full)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="print every row")
    parser.add_argument("--only", choices=("roster", "stats", "moves", "catalogue"),
                        help="just one section")
    args = parser.parse_args()

    dex = load_dex()
    regulation = dex.regulation("m_b")
    species = load("champions_pokemon.json")

    matched = compare_roster(dex, species, regulation, args.full)
    if args.only in (None, "stats"):
        compare_species_fields(dex, species, matched, args.full)
    if args.only in (None, "moves"):
        compare_moves(dex, load("moves.json"), args.full)
    if args.only in (None, "catalogue"):
        compare_catalogue(dex, regulation, args.full)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
