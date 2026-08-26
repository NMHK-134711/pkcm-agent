"""Turn Showdown's Champions mod into a data override layer.

``data/mods/champions/*.ts`` is a table of deltas against the base (Scarlet and
Violet) data. Most entries are one line -- ``isNonstandard: "Past"``, meaning the
move or item simply does not exist in Champions -- and 194 of the moves we would
otherwise treat as legal are exactly that.

This script extracts the **declarative** part of each delta: existence, base
power, accuracy, PP, type, category, priority. What it cannot extract is the
behavioural part, the entries that override a handler function. Those are listed
in the output under ``needs_hand_port`` so they are visible rather than silently
ignored -- reading TypeScript and writing Python is a person's job.

Output: ``data/champions/overrides.json`` (committed).

Usage:
    python scripts/build_champions_overrides.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "data" / "reference" / "champions"
OUT_PATH = ROOT / "data" / "champions" / "overrides.json"

#: Scalar fields worth carrying across. Anything else is behaviour.
SCALAR_FIELDS = (
    "isNonstandard", "basePower", "accuracy", "pp", "priority",
    "type", "category", "target",
)

ENTRY_RE = re.compile(r"^\t([a-z0-9]+): \{", re.M)
#: A line like ``\t\tpp: 5,`` -- one indent level inside the entry.
FIELD_RE = r"^\t\t{field}: (.+?),\s*$"
#: A method definition, ``\t\tonDamage(...) {``: this delta carries behaviour.
METHOD_RE = re.compile(r"^\t\t(\w+)\(", re.M)


def split_entries(text: str) -> dict[str, str]:
    positions = [(m.group(1), m.start()) for m in ENTRY_RE.finditer(text)]
    entries = {}
    for index, (key, start) in enumerate(positions):
        end = positions[index + 1][1] if index + 1 < len(positions) else len(text)
        entries[key] = text[start:end]
    return entries


def parse_value(raw: str):
    raw = raw.strip()
    if raw == "true":
        return True
    if raw == "false":
        return False
    if raw == "null":
        return None
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    if raw.startswith("'") and raw.endswith("'"):
        return raw[1:-1]
    try:
        return int(raw)
    except ValueError:
        return raw


def extract(body: str) -> tuple[dict, list[str]]:
    fields = {}
    for name in SCALAR_FIELDS:
        match = re.search(FIELD_RE.format(field=name), body, re.M)
        if match:
            fields[name] = parse_value(match.group(1))
    methods = sorted({m.group(1) for m in METHOD_RE.finditer(body)})
    return fields, methods


def build(kind: str) -> dict:
    path = REFERENCE_DIR / f"{kind}.ts"
    if not path.exists():
        raise SystemExit(f"missing {path}; run scripts/fetch_showdown_source.py first")

    changes: dict[str, dict] = {}
    needs_hand_port: dict[str, list[str]] = {}
    for key, body in split_entries(path.read_text(encoding="utf-8")).items():
        fields, methods = extract(body)
        if fields:
            changes[key] = fields
        if methods:
            needs_hand_port[key] = methods
    return {"changes": changes, "needs_hand_port": needs_hand_port}


def main() -> int:
    payload = {
        "source": "https://github.com/smogon/pokemon-showdown data/mods/champions (MIT)",
        "note": (
            "Declarative deltas only. isNonstandard 'Past' means the entry does not "
            "exist in Champions; null means Champions re-enables something the base "
            "data marks non-standard. Behavioural overrides are listed under "
            "needs_hand_port and must be ported by reading the TypeScript."
        ),
    }
    for kind in ("moves", "items", "abilities", "conditions"):
        payload[kind] = build(kind)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")

    for kind in ("moves", "items", "abilities", "conditions"):
        changes = payload[kind]["changes"]
        removed = sum(1 for v in changes.values() if v.get("isNonstandard") == "Past")
        enabled = sum(1 for v in changes.values() if "isNonstandard" in v and v["isNonstandard"] is None)
        other = len(changes) - removed - enabled
        print(f"{kind:11} {len(changes):4} deltas  "
              f"({removed} removed, {enabled} re-enabled, {other} altered)  "
              f"{len(payload[kind]['needs_hand_port'])} need hand-porting")
    print(f"\nwritten -> {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
