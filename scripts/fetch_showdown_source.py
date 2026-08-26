"""Download Pokemon Showdown's TypeScript sources as our mechanics reference.

The client JSON we fetch elsewhere carries only numbers. These files carry the
*implementations* -- and, decisively, ``data/mods/champions/`` carries Showdown's
implementation of Pokemon Champions itself: which moves and items the game
actually has, how its status conditions differ from the mainline series, and how
its stats are computed.

MIT licensed (Copyright 2011-2026 Guangcong Luo and contributors).

We do not execute or auto-translate any of this. Declarative deltas are parsed
into an override layer by ``build_champions_overrides.py``; behavioural changes
are read by a human and ported by hand.

Output lands in ``data/reference/`` (gitignored).

Usage:
    python scripts/fetch_showdown_source.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

RAW_BASE = "https://raw.githubusercontent.com/smogon/pokemon-showdown/master/"
REFERENCE_DIR = Path(__file__).resolve().parents[1] / "data" / "reference"
USER_AGENT = "pkcm-agent/0.1 (personal research project)"

#: Base implementations, shared with Scarlet/Violet.
BASE_FILES = ("abilities", "items", "conditions", "moves", "typechart", "natures")

#: Champions' own deltas. Small files, and the only authority we have on how
#: Champions differs from the mainline games.
CHAMPIONS_FILES = ("abilities", "items", "conditions", "moves", "rulesets", "scripts")


def download(path: str) -> str:
    request = urllib.request.Request(RAW_BASE + path, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read().decode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    (REFERENCE_DIR / "champions").mkdir(parents=True, exist_ok=True)
    manifest_path = REFERENCE_DIR / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    targets = [(f"data/{name}.ts", REFERENCE_DIR / f"{name}.ts") for name in BASE_FILES]
    targets += [
        (f"data/mods/champions/{name}.ts", REFERENCE_DIR / "champions" / f"{name}.ts")
        for name in CHAMPIONS_FILES
    ]

    for remote, local in targets:
        if local.exists() and not args.force:
            print(f"skip     {remote}")
            continue
        text = download(remote)
        local.write_text(text, encoding="utf-8")
        manifest[remote] = {
            "bytes": len(text),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        print(f"fetched  {remote:40} {len(text) / 1000:8.1f} KB")

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nmanifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
