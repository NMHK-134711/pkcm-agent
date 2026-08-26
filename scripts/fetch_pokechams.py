"""Download the 포케챔스 dex data -- the only public Champions learnset table.

https://pokemon.yodams.com is a Korean Champions dex. It is a Flutter web app,
so nothing useful is in the HTML; the data ships as static JSON assets under
``/assets/assets/data/``, which is what this fetches.

Why it matters: **``champions_pokemon.json`` carries ``learnableMoveIds``.**
Everything else we have builds learnsets from Showdown's all-generations union,
which is wrong in the permissive direction -- it lets a team builder pick move
combinations that do not exist in Champions. This is the first source that says
what each Pokemon can actually learn.

It also cross-checks a great deal we had only inferred: every move's PP,
priority and spread flag, which species are in, and which moves, abilities and
items exist at all.

Treat it as a *second opinion*, not as the answer key. Showdown's champions mod
stays the mechanics specification (it has the handler code); this is a data
source, and where the two disagree that disagreement is worth reading rather
than silently resolving. ``scripts/compare_pokechams.py`` prints the diff.

Output goes to ``data/raw/pokechams/`` (gitignored), with a MANIFEST recording
url, size and sha256 so a stale cache is detectable.

Usage:
    python scripts/fetch_pokechams.py [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = "https://pokemon.yodams.com/assets/assets/data/"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "pokechams"
USER_AGENT = "pkcm-agent/0.1 (personal project; one-off dex snapshot)"

#: The assets we actually read. The site ships several more -- usage stats,
#: recommended parties, entry-pick priors -- which are opinions about the
#: metagame rather than facts about the game, and no business of the engine.
SOURCES = (
    "champions_pokemon.json",
    "moves.json",
    "abilities.json",
    "items.json",
)


def download(filename: str) -> bytes:
    request = urllib.request.Request(
        BASE_URL + filename, headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the file is already here")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}

    for filename in SOURCES:
        target = RAW_DIR / filename
        if target.exists() and not args.force:
            payload = target.read_bytes()
            print(f"  {filename:28} cached ({len(payload):,} bytes)")
        else:
            payload = download(filename)
            # Parse before writing: a 200 that is really an error page should
            # not land in the cache looking like data.
            parsed = json.loads(payload)
            if not isinstance(parsed, list) or not parsed:
                print(f"  {filename:28} FAILED -- not a non-empty array", file=sys.stderr)
                return 1
            target.write_bytes(payload)
            print(f"  {filename:28} {len(parsed):>5} entries ({len(payload):,} bytes)")

        manifest[filename] = {
            "url": BASE_URL + filename,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    manifest["_fetched"] = datetime.now(timezone.utc).isoformat()
    (RAW_DIR / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nwrote {RAW_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
