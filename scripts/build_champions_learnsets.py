"""Turn the 포케챔스 dex snapshot into the Champions learnset table.

This closes the last big accuracy gap. Everything else we have derives learnsets
from Showdown's ``learnsets.json``, which is an all-generations record of the
main series -- the wrong table for Champions in *both* directions:

  * too permissive: Clefable keeps Soft-Boiled, Seismic Toss, Zap Cannon and the
    rest of the Gen 1-3 TM list, none of which Champions offers.
  * too strict: it misses egg and tutor moves the game does give out -- Tickle,
    Wish, Yawn, Destiny Bond and a few hundred more.

``champions_pokemon.json`` carries ``learnableMoveIds`` per species, which is
the table itself rather than an approximation of it. Every id in it maps onto a
move we already have, which is the check worth trusting: a table full of ids we
could not resolve would mean we had misread the format.

Output: ``data/champions/learnsets.json``, committed, keyed by our species id.

Usage:
    python scripts/fetch_pokechams.py && python scripts/build_champions_learnsets.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pkcm.data.dex import load_dex  # noqa: E402

RAW_DIR = ROOT / "data" / "raw" / "pokechams"
OUT_PATH = ROOT / "data" / "champions" / "learnsets.json"


def normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


#: Their slug spellings our ids do not reach by stripping hyphens. Written out
#: rather than guessed at, because a silent miss leaves that species holding the
#: old over-permissive learnset -- the exact wrongness this script exists to end.
SLUG_ALIASES = {
    "meowstic-female": "meowsticf",
    "meowstic-mega": "meowsticmmega",
    "lycanroc-midday": "lycanroc",
    "basculegion-female": "basculegionf",
}


def our_species_id(slug: str, known: set[str]) -> str | None:
    """Their slug as our species id.

    Three mechanical shapes plus the table above: ``arcanine-hisui`` only loses
    its hyphens, ``sceptile-mega-sceptile`` says the name twice, and the Paldean
    Tauros carry a ``-breed`` suffix we do not use.
    """
    if slug in SLUG_ALIASES:
        return SLUG_ALIASES[slug]
    candidates = [slug.replace("-", "")]
    if "-mega-" in slug:
        candidates.append(slug.split("-mega-")[0].replace("-", "") + "mega")
    if slug.endswith("-breed"):
        candidates.append(slug[: -len("-breed")].replace("-", ""))
    return next((c for c in candidates if c in known), None)


def load(filename: str) -> list[dict]:
    path = RAW_DIR / filename
    if not path.exists():
        raise SystemExit(f"missing {path}\nrun: python scripts/fetch_pokechams.py")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    dex = load_dex()
    species_by_id = {species.id: species for species in dex.species.values()}
    move_by_name = {normalise(move.name): move.id for move in dex.moves.values()}

    their_moves = {record["id"]: record for record in load("moves.json")}
    table: dict[str, list[str]] = {}
    unmapped_species: list[str] = []
    unmapped_moves: set[str] = set()

    for entry in load("champions_pokemon.json"):
        if not entry.get("allowedInChampions"):
            continue
        species_id = our_species_id(entry["slug"], set(species_by_id))
        if species_id is None:
            unmapped_species.append(entry["slug"])
            continue

        learnable: set[str] = set()
        for move_id in entry.get("learnableMoveIds", []):
            record = their_moves.get(move_id)
            if record is None:
                unmapped_moves.add(f"id:{move_id}")
                continue
            ours = move_by_name.get(normalise(record["nameEn"]))
            if ours is None:
                unmapped_moves.add(record["nameEn"])
                continue
            learnable.add(ours)
        table[species_id] = sorted(learnable)

    if unmapped_moves:
        print(f"WARNING: {len(unmapped_moves)} moves did not map: "
              f"{sorted(unmapped_moves)[:10]}", file=sys.stderr)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(table, indent=0, sort_keys=True), encoding="utf-8")

    if not args.quiet:
        total = sum(len(moves) for moves in table.values())
        print(f"  species        {len(table):>6}")
        print(f"  move entries   {total:>6}")
        print(f"  average        {total / max(1, len(table)):>6.1f} per species")
        print(f"\nwrote {OUT_PATH.relative_to(ROOT)}")

    if unmapped_species:
        # Not a warning. A species we fail to map keeps the all-generations
        # learnset -- exactly the wrongness this file exists to replace -- and it
        # would keep it without saying anything.
        print(f"\nFAILED: {len(unmapped_species)} allowed species did not map to "
              f"an id we know: {unmapped_species}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
