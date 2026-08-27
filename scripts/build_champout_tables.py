"""Build the committed tables from the ROM dumps.

``data/raw/champout/`` is gitignored, and the engine cannot depend on a cache
that may not be there, so what the ROM says gets written down here:

``data/champions/moves_available.json``  which moves exist in Champions at all
``data/champions/learnsets.json``        which species learns which

Both were previously answered by someone other than the game. Move existence
came from Showdown's ``isNonstandard`` flag, which is Showdown's judgement about
its own dex -- and it was wrong three times. Learnsets came from the 포케챔스
dex, which is a Korean Champions dex and much closer, and is still short 252
entries across 160 species.

**This is the ordering docs/HANDOFF.md now states outright**: the ROM is the
game's tables, and everything else is somebody reading the game.

Both outputs are diffed against what they replace, because a table that changed
silently is a table nobody checked.

Usage:
    python scripts/fetch_champout.py
    python scripts/build_champout_tables.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pkcm.data import champout  # noqa: E402
from pkcm.data.dex import load_dex  # noqa: E402

OUT_DIR = ROOT / "data" / "champions"
AVAILABLE_PATH = OUT_DIR / "moves_available.json"
LEARNSETS_PATH = OUT_DIR / "learnsets.json"


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):  # pragma: no cover
            pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="report the diff without writing")
    args = parser.parse_args()

    dex = load_dex()
    regulation = dex.regulation("m_b")
    fieldable = set(regulation.legal_species) | set(regulation.legal_megas)
    try:
        rom = champout.load(dex, fieldable)
    except champout.MissingDump as error:
        print(error)
        return 1

    names = json.loads((OUT_DIR / "names.json").read_text(encoding="utf-8"))

    def move_name(move_id: str) -> str:
        return names["moves"].get(move_id, move_id)

    # -- which moves exist ------------------------------------------------- #
    available = sorted(move_id for move_id, yes in rom.available.items() if yes)
    was = {move.id for move in dex.moves.values()
           if move.raw.get("isNonstandard") is None}
    gone = sorted(was - set(available))
    gained = sorted(set(available) - was)
    print(f"moves: {len(available)} exist per the ROM, {len(was)} per Showdown's "
          f"isNonstandard")
    print(f"  dropped: {[move_name(m) for m in gone]}")
    print(f"  added  : {[move_name(m) for m in gained]}")

    # -- who learns what --------------------------------------------------- #
    previous = json.loads(LEARNSETS_PATH.read_text(encoding="utf-8")) \
        if LEARNSETS_PATH.exists() else {}
    learnsets = {species: sorted(moves)
                 for species, moves in rom.learnset.items() if moves}

    extra: Counter = Counter()
    short: Counter = Counter()
    changed = 0
    for species, moves in learnsets.items():
        before = set(previous.get(species, ()))
        if not before:
            continue
        after = set(moves)
        if before != after:
            changed += 1
        for move_id in before - after:
            extra[move_id] += 1
        for move_id in after - before:
            short[move_id] += 1

    dropped_species = sorted(set(previous) - set(learnsets))
    print(f"\nlearnsets: {len(learnsets)} species from the ROM, "
          f"{len(previous)} in the file being replaced")
    print(f"  {changed} species change")
    print(f"  {sum(short.values())} entries the ROM adds; most common: "
          f"{[(move_name(m), c) for m, c in short.most_common(5)]}")
    print(f"  {sum(extra.values())} entries the ROM removes; most common: "
          f"{[(move_name(m), c) for m, c in extra.most_common(5)]}")
    if dropped_species:
        print(f"  {len(dropped_species)} species lose their row entirely: "
              f"{[names['species'].get(s, s) for s in dropped_species[:8]]}")

    missing = sorted(fieldable - set(learnsets))
    if missing:
        print(f"\n  !! {len(missing)} fieldable species have no ROM row: "
              f"{[names['species'].get(s, s) for s in missing]}")
        print("     their existing rows are kept rather than dropped")
        for species in missing:
            if species in previous:
                learnsets[species] = previous[species]

    if args.dry_run:
        print("\n(dry run -- nothing written)")
        return 0

    AVAILABLE_PATH.write_text(json.dumps(available, indent=0), encoding="utf-8")
    LEARNSETS_PATH.write_text(
        json.dumps(dict(sorted(learnsets.items())), indent=0, sort_keys=True),
        encoding="utf-8")
    print(f"\nwrote {AVAILABLE_PATH}")
    print(f"wrote {LEARNSETS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
