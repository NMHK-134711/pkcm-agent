"""Diff our dex against the 포케챔스 snapshot, and say where they disagree.

Two independent readings of the same game: ours is built from Showdown's
champions mod, theirs from the game's own dex. Neither is the answer key --
the game is -- so this prints the disagreements rather than resolving them.

Matching is by **English name**, not by id. The two sources number things
differently and Champions renumbered about thirty moves on top of that, so an
id match would be a coincidence and a name match is a fact.

Usage:
    python scripts/compare_pokechams.py            # summary
    python scripts/compare_pokechams.py --learnsets  # per-species move diffs
    python scripts/compare_pokechams.py --moves      # per-move field diffs
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pkcm.data.dex import load_dex  # noqa: E402

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "pokechams"


def normalise(name: str) -> str:
    """Both sources spell names for humans; ids are what we compare on."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def load(filename: str) -> list[dict]:
    path = RAW_DIR / filename
    if not path.exists():
        raise SystemExit(f"missing {path}\nrun: python scripts/fetch_pokechams.py")
    return json.loads(path.read_text(encoding="utf-8"))


def champions_max_pp(base_pp: int) -> int:
    from pkcm.engine.pokemon import max_pp

    return max_pp(base_pp)


def build_move_index(their_moves: list[dict]) -> dict[str, dict]:
    """Their numeric move id -> the move record."""
    return {move["id"]: move for move in their_moves}


def section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def their_species_id(entry: dict) -> str:
    """Their ``slug`` is our species id with hyphens: ``arcanine-hisui``."""
    return entry["slug"].replace("-", "")


def compare_roster(dex, theirs, regulation) -> None:
    section("Roster")
    their_in = {their_species_id(p): p for p in theirs if p.get("allowedInChampions")}
    ours = {s: s for s in regulation.legal_species | regulation.legal_megas}

    print(f"  theirs: {len(their_in):>4} allowed   ours: {len(ours):>4} legal species")
    only_theirs = sorted(set(their_in) - set(ours))
    only_ours = sorted(set(ours) - set(their_in))
    if only_theirs:
        print(f"  only in pokechams ({len(only_theirs)}): "
              f"{[their_in[k]['nameEn'] for k in only_theirs[:14]]}")
    if only_ours:
        print(f"  only in ours      ({len(only_ours)}): "
              f"{[dex.species[k].name for k in only_ours[:14]]}")
    if not only_theirs and not only_ours:
        print("  identical")


def compare_moves(dex, their_moves, verbose: bool) -> None:
    section("Moves")
    their_in = {normalise(m["nameEn"]): m for m in their_moves
                if m.get("allowedInChampions")}
    ours = {normalise(m.name): m for m in dex.moves.values()
            if dex.exists_in_champions(m)}

    print(f"  theirs: {len(their_in):>4} allowed   ours: {len(ours):>4} in Champions")
    only_theirs = sorted(set(their_in) - set(ours))
    only_ours = sorted(set(ours) - set(their_in))
    if only_theirs:
        print(f"  only in pokechams: {[their_in[k]['nameEn'] for k in only_theirs]}")
    if only_ours:
        print(f"  only in ours    : {[ours[k].name for k in only_ours]}")

    shared = sorted(set(their_in) & set(ours))
    mismatched = {"pp": [], "power": [], "accuracy": [], "priority": [], "spread": []}
    for key in shared:
        mine, yours = ours[key], their_in[key]
        # Their ``pp`` is the number the game shows, which is the Champions
        # max -- not the base PP our data carries. Compare like with like.
        if champions_max_pp(mine.pp) != yours["pp"]:
            mismatched["pp"].append((mine.name, champions_max_pp(mine.pp), yours["pp"]))
        if mine.category != "Status" and mine.base_power and mine.base_power != yours["power"]:
            mismatched["power"].append((mine.name, mine.base_power, yours["power"]))
        their_acc = yours["accuracy"]
        mine_acc = 100 if mine.accuracy is None else mine.accuracy
        if their_acc and mine_acc != their_acc:
            mismatched["accuracy"].append((mine.name, mine_acc, their_acc))
        if mine.priority != yours["priority"]:
            mismatched["priority"].append((mine.name, mine.priority, yours["priority"]))
        from pkcm.engine.moves import SPREAD_TARGETS

        mine_spread = mine.target in SPREAD_TARGETS
        if mine_spread != bool(yours["isSpread"]):
            mismatched["spread"].append((mine.name, mine_spread, yours["isSpread"]))

    for field, rows in mismatched.items():
        print(f"  {field:9} {len(rows):>4} disagree" + (" " if rows else "  (clean)"))
        if rows and verbose:
            for name, mine, yours in rows[:40]:
                print(f"      {name:22} ours={mine!s:>6}  theirs={yours!s:>6}")
            if len(rows) > 40:
                print(f"      ... and {len(rows) - 40} more")


def compare_learnsets(dex, theirs, their_moves, regulation, verbose: bool) -> None:
    section("Learnsets  (the reason this source exists)")
    by_id = build_move_index(their_moves)
    ours_by_id = {s.id: s.id for s in dex.species.values()}

    from pkcm.engine.legality import learnable_moves

    total_theirs = total_ours = total_extra = 0
    worst: list[tuple[int, str, list[str]]] = []
    missing_species = []

    for entry in theirs:
        if not entry.get("allowedInChampions"):
            continue
        species_id = ours_by_id.get(their_species_id(entry))
        if species_id is None or species_id not in regulation.legal_species:
            missing_species.append(entry["nameEn"])
            continue

        their_set = set()
        for move_id in entry.get("learnableMoveIds", []):
            record = by_id.get(move_id)
            if record is not None:
                their_set.add(normalise(record["nameEn"]))

        our_set = {normalise(dex.moves[m].name) for m in learnable_moves(dex, species_id)}
        total_theirs += len(their_set)
        total_ours += len(our_set)
        extra = our_set - their_set
        total_extra += len(extra)
        if extra:
            worst.append((len(extra), entry["nameEn"],
                          sorted(dex.moves[m].name for m in learnable_moves(dex, species_id)
                                 if normalise(dex.moves[m].name) in extra)))

    print(f"  species compared      : {len([e for e in theirs if e.get('allowedInChampions')]) - len(missing_species)}")
    print(f"  moves they list       : {total_theirs:,}")
    print(f"  moves we allow        : {total_ours:,}")
    print(f"  we allow but they do not: {total_extra:,}"
          f"  ({total_extra / max(1, total_ours):.0%} of ours)")
    if missing_species:
        print(f"  unmatched species     : {len(missing_species)} {missing_species[:8]}")

    worst.sort(reverse=True)
    print("\n  most over-permissive species:")
    for count, name, moves in worst[:15]:
        shown = ", ".join(moves[:8]) + (" ..." if len(moves) > 8 else "")
        print(f"    {name:18} +{count:<4} {shown}")
    if verbose:
        print("\n  full list:")
        for count, name, moves in worst:
            print(f"    {name}: {', '.join(moves)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learnsets", action="store_true", help="every species' diff")
    parser.add_argument("--moves", action="store_true", help="every disagreeing field")
    args = parser.parse_args()

    dex = load_dex()
    regulation = dex.regulation("m_b")

    compare_roster(dex, load("champions_pokemon.json"), regulation)
    compare_moves(dex, load("moves.json"), args.moves)
    compare_learnsets(dex, load("champions_pokemon.json"), load("moves.json"),
                      regulation, args.learnsets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
