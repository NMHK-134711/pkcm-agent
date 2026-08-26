"""Download Pokemon Showdown client data and normalize it to plain JSON.

Showdown serves two flavors under https://play.pokemonshowdown.com/data/ :
  * ``*.json``  - already valid JSON.
  * ``*.js``    - ``exports.BattleX = { ... };`` with unquoted keys (JSON5).

Both are *data only* (the client build strips the handler functions), which is
exactly what we want: we implement the mechanics ourselves and only need the
numbers and the metadata.

Output goes to ``data/raw/`` (gitignored). ``data/raw/MANIFEST.json`` records the
URL, byte size and sha256 of every download so a stale cache is detectable --
upstream mutates these files silently as the metagame changes.

Usage:
    python scripts/fetch_showdown_data.py [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = "https://play.pokemonshowdown.com/data/"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
USER_AGENT = "pkcm-agent/0.1 (personal research project)"

# remote filename -> (local stem, exports.<name> for .js files)
SOURCES: dict[str, tuple[str, str | None]] = {
    "pokedex.json": ("pokedex", None),
    "moves.json": ("moves", None),
    "learnsets.json": ("learnsets", None),
    "abilities.js": ("abilities", "BattleAbilities"),
    "items.js": ("items", "BattleItems"),
    "typechart.js": ("typechart", "BattleTypeChart"),
    "formats-data.js": ("formats_data", "BattleFormatsData"),
    "aliases.js": ("aliases", "BattleAliases"),
}


def download(filename: str) -> bytes:
    request = urllib.request.Request(BASE_URL + filename, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def parse(raw: bytes, exports_name: str | None) -> dict:
    text = raw.decode("utf-8")
    if exports_name is None:
        return json.loads(text)

    prefix = f"exports.{exports_name} = "
    if not text.startswith(prefix):
        raise ValueError(f"expected {prefix!r} at start of payload, got {text[:60]!r}")
    body = text[len(prefix):].strip().rstrip(";")

    import json5  # only needed for the JSON5-flavored .js payloads

    return json5.loads(body)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW_DIR / "MANIFEST.json"
    manifest: dict[str, dict] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for filename, (stem, exports_name) in SOURCES.items():
        out_path = RAW_DIR / f"{stem}.json"
        if out_path.exists() and not args.force:
            print(f"skip     {stem:14} (already present, use --force to refresh)")
            continue

        raw = download(filename)
        data = parse(raw, exports_name)
        out_path.write_text(json.dumps(data, indent=0, sort_keys=True), encoding="utf-8")

        manifest[stem] = {
            "url": BASE_URL + filename,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "entries": len(data),
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        print(f"fetched  {stem:14} {len(raw) / 1e6:6.2f} MB  {len(data):5} entries")

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nmanifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
