"""Download the champout dumps -- Champions' own tables, out of the Switch ROM.

https://github.com/projectpokemon/champout is a text dump of Pokemon Champions
on Switch: the master data the game ships and downloads, plus parses of the
personal and learnset tables.

**This outranks every other source this repo has.** Showdown's champions mod is
a mechanics reference written by people reading the game; the 포케챔스 dex and
the pkmnchamps archive are Korean sites reporting it. This is the game's tables.
The rule that fell out of that ordering, and cost two wrong clauses before it
was written down, is in docs/HANDOFF.md: **Showdown is not a rules source.**

What each file is:

``masterdata/personal.json``   per-species: base stats, types, abilities
``masterdata/waza.json``       per-move: power, accuracy, PP, priority
``masterdata/waza_learn.json`` **which species learns which move**
``masterdata/item.json``       the item table
``parse/move_availability.txt``  which moves exist at all, by national number
``parse/species_with_move.txt``  the learnset again, grouped by move, in
                                 Showdown's English naming -- which is what
                                 makes it cheap to line up against our ids
``parse/personal_dump.txt``      the personal table, human-readable

Output goes to ``data/raw/champout/`` (gitignored), with a MANIFEST recording
url, size and sha256, so a stale cache is detectable.

Usage:
    python scripts/fetch_champout.py [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/projectpokemon/champout/main/"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "champout"
USER_AGENT = "pkcm-agent/0.1 (personal project; one-off ROM table snapshot)"

SOURCES = (
    "masterdata/personal.json",
    "masterdata/waza.json",
    "masterdata/waza_learn.json",
    "masterdata/item.json",
    "parse/move_availability.txt",
    "parse/species_with_move.txt",
    "parse/personal_dump.txt",
)


def download(path: str) -> bytes:
    request = urllib.request.Request(
        BASE_URL + path, headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the file is already here")
    args = parser.parse_args()

    manifest: dict[str, dict] = {}
    failures: list[str] = []

    for path in SOURCES:
        target = RAW_DIR / Path(path).name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not args.force:
            payload = target.read_bytes()
            print(f"  {Path(path).name:26} cached ({len(payload):,} bytes)")
        else:
            try:
                payload = download(path)
            except Exception as error:  # pragma: no cover - network
                print(f"  {Path(path).name:26} FAILED: {error}")
                failures.append(path)
                continue
            target.write_bytes(payload)
            print(f"  {Path(path).name:26} fetched ({len(payload):,} bytes)")
        manifest[Path(path).name] = {
            "url": BASE_URL + path,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    manifest["_fetched"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (RAW_DIR / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"\nmanifest -> {RAW_DIR / 'MANIFEST.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
