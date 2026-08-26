"""Group the roster's abilities by the shape of their Showdown implementation.

201 distinct abilities appear on the M-B roster, but far fewer distinct *shapes*:
Blaze, Torrent, Overgrow and Swarm are one implementation with the type swapped;
Limber, Immunity and Insomnia are one implementation with the status swapped.
Porting shape by shape instead of ability by ability is the difference between
reading four files and reading two hundred.

Reports, for each group of abilities sharing a handler signature:
  * which handlers they use
  * how many roster species carry them
  * whether we have implemented them yet

Usage:
    python scripts/ability_report.py                 # groups, largest first
    python scripts/ability_report.py --show intimidate levitate
    python scripts/ability_report.py --todo          # only what is unimplemented
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REFERENCE = ROOT / "data" / "reference" / "abilities.ts"
CHAMPIONS_REFERENCE = ROOT / "data" / "reference" / "champions" / "abilities.ts"

ENTRY_RE = re.compile(r"^\t([a-z0-9]+): \{", re.M)
HANDLER_RE = re.compile(r"^\t\t(on[A-Za-z]+)\s*[(:]", re.M)


def split_entries(text: str) -> dict[str, str]:
    positions = [(m.group(1), m.start()) for m in ENTRY_RE.finditer(text)]
    entries = {}
    for index, (key, start) in enumerate(positions):
        end = positions[index + 1][1] if index + 1 < len(positions) else len(text)
        entries[key] = text[start:end]
    return entries


def roster_abilities() -> collections.Counter:
    from pkcm.data.dex import load_dex

    dex = load_dex()
    regulation = dex.regulation("m_b")
    counts: collections.Counter = collections.Counter()
    for species_id in regulation.legal_species | regulation.legal_megas:
        for ability in dex.species[species_id].abilities:
            counts[ability] += 1
    return counts


def implemented() -> set[str]:
    from pkcm.engine import effects

    try:
        from pkcm.engine import abilities  # noqa: F401  -- registers on import
    except ImportError:
        pass
    return set(effects.registered("ability"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", nargs="*", help="print these abilities' source verbatim")
    parser.add_argument("--todo", action="store_true", help="only unimplemented abilities")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    if not REFERENCE.exists():
        print(f"missing {REFERENCE}; run scripts/fetch_showdown_source.py", file=sys.stderr)
        return 1

    entries = split_entries(REFERENCE.read_text(encoding="utf-8"))
    # The Champions mod mostly stubs entries with `inherit: true`; naively
    # overwriting would hide the base implementation those stubs inherit.
    mod = split_entries(CHAMPIONS_REFERENCE.read_text(encoding="utf-8"))
    for key, body in mod.items():
        entries[key] = entries.get(key, "") + "\n// --- champions mod override ---\n" + body

    if args.show:
        for name in args.show:
            body = entries.get(name)
            print(body if body else f"// {name}: not found\n")
        return 0

    counts = roster_abilities()
    done = implemented()

    groups: dict[tuple[str, ...], list[str]] = collections.defaultdict(list)
    for ability in counts:
        body = entries.get(ability, "")
        signature = tuple(sorted({m.group(1) for m in HANDLER_RE.finditer(body)}))
        groups[signature].append(ability)

    ordered = sorted(
        groups.items(),
        key=lambda item: -sum(counts[a] for a in item[1]),
    )

    total_slots = sum(counts.values())
    covered_slots = sum(counts[a] for a in counts if a in done)
    print(f"roster abilities: {len(counts)} distinct, {total_slots} species slots")
    print(f"implemented:      {len(done & set(counts))} distinct, "
          f"{covered_slots} slots ({100 * covered_slots / total_slots:.0f}%)")
    print(f"distinct handler shapes: {len(groups)}\n")

    for signature, abilities in ordered[: args.limit]:
        remaining = [a for a in abilities if a not in done]
        if args.todo and not remaining:
            continue
        slots = sum(counts[a] for a in abilities)
        label = ", ".join(signature) if signature else "(no handlers -- pure data or engine-side)"
        print(f"  [{slots:3} slots] {label}")
        shown = sorted(remaining if args.todo else abilities, key=lambda a: -counts[a])
        print(f"      {', '.join(f'{a}({counts[a]})' for a in shown[:14])}"
              + (" ..." if len(shown) > 14 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
